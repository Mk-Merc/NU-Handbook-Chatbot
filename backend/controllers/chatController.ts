import { Request, Response } from 'express';
import fs from 'fs';
import dotenv from 'dotenv';
import { VectorChunk, OLLAMA_URL, VECTOR_PATH, getHandbookChunks, getCleanHandbookText, getEmbedding } from '../services/vectorService';
import natural from 'natural';
import { similarity } from 'ml-distance';

dotenv.config();

const CHAT_MODEL = 'qwen3:1.7b';

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

// Finds the closest matching words in the handbook vocabulary
const expandQueryWithFuzzyMatching = (cleanQuery: string) => {
    // Set = list of unique words from the handbook text
    const handbookVocabulary = new Set(getCleanHandbookText().toLowerCase().split(/\W+/));
    const queryWords = cleanQuery.split(' ');

    const fuzzyWords = queryWords.map((word: string) => {
        if (word.length > 5 && !handbookVocabulary.has(word)) {
            let closestWord = word;
            let highestScore = 0;

            handbookVocabulary.forEach((vocabWord) => {
                // Calculate the similarity score using Jaro-Winkler distance
                const score = natural.JaroWinklerDistance(word, vocabWord);

                if (score > highestScore) {
                    highestScore = score;
                    closestWord = vocabWord;
                }
            });

            if (highestScore > 0.8) {
                return closestWord;
            }
        }
        return word;
    });
    return fuzzyWords.join(' ');
};

// Calculates the similarity between the user's query and every pre-generated vector
const getVectorScores = async (expandedQuery: string) => {
    const queryEmbedding = await getEmbedding(expandedQuery);

    // item = each chunk and embedding in the vectorStore
    return vectorStore.map((item, index) => ({ 
        index, 
        score: similarity.cosine(queryEmbedding, item.embedding) 
    }))
    .sort((itemA, itemB) => itemB.score - itemA.score);
};

// Scores keywords from user's query based on how frequent and unique they are against the tf-idf index
const getTFIDFScores = (expandedQuery: string) => {
    const tfIDFScores: { index: number, score: number }[] = [];

    tfIDF.tfidfs(expandedQuery, (index, measure) => tfIDFScores.push({ index: index, score: measure }));

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

// Retrieves the chunks and their surrounding context
const expandWindow = (fusedResults: {index: number, score: number}[]) => {
    // Take the top 3 results
    const topMatches = fusedResults.slice(0, 3);
    const contextChunks: string[] = [];

    topMatches.forEach(match => {
        for (let index = 2; index > 0; index--) {
            if (match.index - index >= 0) contextChunks.push(chunks[match.index - index]);
        }
        contextChunks.push(chunks[match.index]);

        for (let index = 1; index <= 2; index++) {
            if (match.index + index < chunks.length) contextChunks.push(chunks[match.index + index]);
        }
    });
    // ... = flattens the contextChunks array into this new array
    return [...new Set(contextChunks)].join('\n\n---\n\n');
};

export const handleChat = async (request: Request, respond: Response) => {
    const { messages } = request.body;

    if (!messages || messages.length === 0) return respond.status(400).json({ error: "Messages are required" });

    const latestMessage = messages[messages.length - 1].content;

    try {
        const cleanQuery = latestMessage.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(' ')
            .filter((word: string) => word.length > 2 && !stopWords.has(word))
            .join(' ');

        const expandedQuery = expandQueryWithFuzzyMatching(cleanQuery);
        const vectorScores = await getVectorScores(expandedQuery);
        const tfIDFScores = getTFIDFScores(expandedQuery);
        const fusedResults = fuseScores(vectorScores, tfIDFScores);
        const context = expandWindow(fusedResults);

        const response = await fetch(`${OLLAMA_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CHAT_MODEL,
                stream: false,
                options: { num_predict: 1500, temperature: 0.0 },
                messages: [
                    { role: 'system', 
                      content: `You are a student handbook assistant. Answer the user's question using ONLY the context provided below. 
                                If the answer is not in the context, say "I don't know based on the handbook.
                                "\n\nCRITICAL RULES:\n1. Do NOT make up your own math examples.
                                \n2. The National University grading scale is 1.0 to 4.0. Do NOT use percentages like 90 or 95.
                                \n\nCONTEXT:\n${context}` 
                    },
                    // Ollama API expects the user's message to be in one array
                    ...messages
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