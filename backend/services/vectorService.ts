import fs from 'fs';
import dotenv from 'dotenv';

dotenv.config();

export const OLLAMA_URL = `http://${process.env.OLLAMA_URL}`;
// The specific Ollama model used to convert text into mathematical vectors
const EMBED_MODEL = 'nomic-embed-text';
export const VECTOR_PATH = './data/vectors.json';
const TEXT_PATH = './data/handbook.txt';

interface OllamaResponse {
    embeddings: number[][];
}

export interface VectorChunk {
    chunk: string;
    embedding: number[];
}

// Promise<number[]> = this async function will eventually resolve to an array of numbers
export const getEmbedding = async (text: string): Promise<number[]> => {
    const respond = await fetch(`${OLLAMA_URL}/api/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Turn JS object into JSON string to send to Ollama API
        body: JSON.stringify({ model: EMBED_MODEL, input: text })
    });
    const data = await respond.json() as OllamaResponse;

    return data.embeddings[0];
};

export const getCleanHandbookText = (): string => {
    const rawText = fs.readFileSync(TEXT_PATH, 'utf-8');

    return rawText.replace(/\.{3,}/g, ' ').replace(/\s+/g, ' ').trim();
};

export const getHandbookChunks = (): string[] => {
    const cleanText = getCleanHandbookText();
    
    const chunkSize = 1500;
    const overlapSize = 200;
    const stepSize = chunkSize - overlapSize;
    const chunks: string[] = [];
    
    for (let index = 0; index < cleanText.length; index += stepSize) {
        // substring = start at the current index and stop at the current index + chunkSize
        chunks.push(cleanText.substring(index, index + chunkSize));
    }
    return chunks;
};

const generateVectors = async () => {
    const chunks = getHandbookChunks();
    const vectorStore: VectorChunk[] = [];
    
    console.log(`Embedding ${chunks.length} chunks...`);
    
    for (let index = 0; index < chunks.length; index++) {
        const embedding = await getEmbedding(chunks[index]);

        vectorStore.push({ chunk: chunks[index], embedding: embedding });
        
        if (index % 10 === 0) console.log(`Progress: ${index}/${chunks.length}`);
    }
    // Convert vector array into a JSON string to save it for future use
    fs.writeFileSync(VECTOR_PATH, JSON.stringify(vectorStore));

    console.log(`Saved ${vectorStore.length} vectors to ${VECTOR_PATH}`);
};

// Run if this file is executed directly
if (require.main === module) {
    generateVectors();
}