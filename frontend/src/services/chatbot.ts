export const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

export interface Message {
    id: string;
    sender: 'ME' | 'BOT';
    text: string;
}

interface ChatResponse {
    reply: string;
}

export const sendMessageToBot = async (messages: { role: string, content: string }[]) => {
    const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
    });
    if (!response.ok) throw new Error(`Backend error: ${response.status}`);

    const data = await response.json() as ChatResponse;

    return data.reply; 
};