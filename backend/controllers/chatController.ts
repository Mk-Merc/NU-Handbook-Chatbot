import { Request, Response } from 'express';
import fs from 'fs';
import dotenv from 'dotenv';
import { VectorChunk, OLLAMA_URL, VECTOR_PATH, getHandbookChunks, getEmbedding } from '../services/vectorService';
import natural from 'natural';
import { similarity } from 'ml-distance';
import { AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';

dotenv.config();

const CHAT_MODEL = 'qwen3:1.7b';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

interface OllamaChatResponse {
    message: {
        content: string;
    };
}

// Load the pre-generated vectors
// Turn the JSON string back into JS object to use for similarity calculations
const vectorStore: VectorChunk[] = JSON.parse(fs.readFileSync(VECTOR_PATH, 'utf-8'));

// Load handbook chunks and build the TF-IDF index for keyword search
const chunks = getHandbookChunks();
const tfIDF = new natural.TfIdf();
chunks.forEach(chunk => tfIDF.addDocument(chunk));

const stopWords = new Set([ 'who', 'what', 'the', 'a', 'an', 'is', 'are', 'of', 'in', 'to', 
                            'and', 'for', 'with', 'about', 'how', 'do', 'i', 'can', 'my', 
                            'nu', 'national', 'university', 'become', 'be', 'get', 'make', 
                            'have', 'want', 'need', 'know', 'tell', 'give', 'show', 'please' ]);


let rerankerModel: any = null;
let rerankerTokenizer: any = null;

// Initializes the cross-encoder reranker to deeply score query-chunk pairs
const initializeReranker = async () => {
    try {
        // dtype currently lacks TypeScript support in the library, so the any is used to bypass it
        // q8 = 8-bit quantization (smaller model, faster inference, minimal accuracy loss)
        const rerankerOptions: any = { 
            dtype: 'q8'
        };

        console.log("Loading reranker model...");

        rerankerTokenizer = await AutoTokenizer.from_pretrained('Xenova/bge-reranker-base', rerankerOptions);
        rerankerModel = await AutoModelForSequenceClassification.from_pretrained('Xenova/bge-reranker-base', rerankerOptions);

        console.log("Reranker ready!");
    } catch (error) {
        console.error("Failed to load reranker:", error);
    }
};
initializeReranker();

// Calculates the similarity between the user's query and every pre-generated vector
const getVectorScores = async (cleanQuery: string) => {
    const queryEmbedding = await getEmbedding(cleanQuery);

    // item = each chunk and embedding in the vectorStore
    return vectorStore.map((item, index) => ({ 
        index, 
        score: similarity.cosine(queryEmbedding, item.embedding) 
    }))
    .sort((itemA, itemB) => itemB.score - itemA.score);
};

// Scores keywords from user's query based on how frequent and unique they are against the tf-idf index
const getTFIDFScores = (cleanQuery: string) => {
    const tfIDFScores: { index: number, score: number }[] = [];

    tfIDF.tfidfs(cleanQuery, (index, measure) => tfIDFScores.push({ index: index, score: measure }));

    return tfIDFScores.sort((itemA, itemB) => itemB.score - itemA.score);
};

// Combines vector and tf-idf rankings into a single final score using Reciprocal Rank Fusion
const fuseScores = (vectorScores: {index: number, score: number}[], tfIDFScores: {index: number, score: number}[]) => {
    const rrfScores: { [key: number]: number } = {};
    // 60 is a commonly used constant in RRF to balance the influence of rank and score
    // A higher value smooths the rankings, preventing the #1 result from dominating
    const rrfConstant = 60;
    // rank = the position of the item in the sorted list (0 for highest score, 1 for second highest, etc.)
    // ?? = nullish coalescing operator, returns the right-hand side if the left-hand side is null or undefined
    vectorScores.forEach((item, rank) => rrfScores[item.index] = (rrfScores[item.index] ?? 0) + (1 / (rrfConstant + rank + 1)));
    tfIDFScores.forEach((item, rank) => rrfScores[item.index] = (rrfScores[item.index] ?? 0) + (1 / (rrfConstant + rank + 1)));

    return Object.keys(rrfScores).map(index => ({ 
        index: parseInt(index), 
        score: rrfScores[parseInt(index)] 
    }))
    .sort((itemA, itemB) => itemB.score - itemA.score);
};

// Scores the top candidates by comparing user query and candidate chunk
const rerankContext = async (rerankerQuery: string, candidates: {index: number, text: string}[]): Promise<number> => {
    if (!rerankerModel || !rerankerTokenizer) {
        return candidates[0].index;
    }

    try {
        const texts = candidates.map(candidate => candidate.text);
        
        const inputs = await rerankerTokenizer(
            new Array(texts.length).fill(rerankerQuery), 
            { text_pair: texts, padding: true, truncation: true }
        );

        const { logits } = await rerankerModel(inputs);
        
        const scores = logits.data;
        const scored = candidates.map((candidate, index) => ({
            // Create a new object and copy the original candidate properties (index, text)
            ...candidate,
            // Append the new score generated by the reranker model into this new object
            score: scores[index]
        }));
        scored.sort((itemA, itemB) => itemB.score - itemA.score);

        console.log('Highest Score:', scored[0].score.toFixed(4));
        console.log('Chunk ID:', scored[0].index);
        console.log('Chunk Text:', scored[0].text.substring(0, 1000) + '...');

        return scored[0].index;
    } catch (error) {
        console.error("Reranker failed, falling back to candidates:", error);

        return candidates[0].index;
    }
};

// Retrieves the best chunk and their surrounding context
const expandWindow = (bestChunkIndex: number): string => {
    const contextChunks: string[] = [];

    // Previous chunk
    if (bestChunkIndex - 1 >= 0) contextChunks.push(chunks[bestChunkIndex - 1]);

    contextChunks.push(chunks[bestChunkIndex]);

    // Next chunk
    if (bestChunkIndex + 1 < chunks.length) contextChunks.push(chunks[bestChunkIndex + 1]);

    // ... = flattens the contextChunks array into this new array
    return [...new Set(contextChunks)].join('\n\n---\n\n');
};

// Resolves pronouns in user queries by adding the previous bot reply to the start of the query 
const buildRerankerQuery = (latestMessage: string, messages: Message[]): string => {
    const pronounRegex = /\b(him|her|it|them|they|he|she|his|hers|their)\b/i;
    
    if (pronounRegex.test(latestMessage) && messages.length >= 3) {
        const previousBotReply = messages[messages.length - 2].content;
        const rerankerQuery = `${previousBotReply} ${latestMessage}`;

        console.log('Pronoun detected! Expanding Reranker Query with user previous query.');

        return rerankerQuery;
    } else {
        console.log('Fresh question. Using only current query for Reranker Query.');

        return latestMessage;
    }
};

export const handleChat = async (request: Request, respond: Response) => {
    // The messages includes the system prompt and the bot greeting before the user query
    const { messages } = request.body as { messages: Message[] };

    if (!messages || messages.length === 0) return respond.status(400).json({ error: "Messages are required" });

    const latestMessage = messages[messages.length - 1].content;
    console.log('User Query:', latestMessage);

    try {
        const cleanQuery = latestMessage.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(' ')
            .filter((word: string) => word.length > 2 && !stopWords.has(word))
            .join(' ');

        const vectorScores = await getVectorScores(cleanQuery);
        const tfIDFScores = getTFIDFScores(cleanQuery);
        const fusedResults = fuseScores(vectorScores, tfIDFScores);

        // Get the top 10 candidates
        const candidateChunks = fusedResults.slice(0, 10).map(result => ({
            index: result.index,
            text: chunks[result.index]
        }));

        const rerankerQuery = buildRerankerQuery(latestMessage, messages);
        const bestChunkIndex = await rerankContext(rerankerQuery, candidateChunks);
        const context = expandWindow(bestChunkIndex);

        console.log('Context length:', context.length, 'characters');
        console.log('\n\n\n\n\n');

        const response = await fetch(`${OLLAMA_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CHAT_MODEL,
                stream: false,
                options: { num_predict: 1500, temperature: 0.0 },
                messages: [
                    { role: 'system', 
                      content: `You are a helpful AI assistant for the National University Student Handbook. 
                                Read the context provided below carefully and answer the user's question based on it.
                                If the user asks a question, quote the exact text from the context that answers it.
                                If the answer is truly not in the context, say "I couldn't find that in the handbook."
                                CONTEXT:${context}` 
                    },
                    // Ollama API expects the user's message to be in one array
                    // Grabs the last two items in the array (previous bot reply and current user query)
                    ...messages.slice(-2)
                ]
            })
        });
        const data = await response.json() as OllamaChatResponse;

        respond.json({ reply: data.message.content });
    } catch (error) {
        console.error("Network Error:", error);

        respond.status(500).json({ error: "Failed to connect to Ollama" });
    }
};