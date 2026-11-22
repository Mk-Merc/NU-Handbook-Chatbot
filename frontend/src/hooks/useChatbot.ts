import { useState } from 'react';
import { Message, sendMessageToBot } from '../services/chatbot';

export const useChatbot = () => {
    const [messages, setMessages] = useState<Message[]>([
        { id: '1', text: 'Hello! I am your NU Handbook Chatbot. How can I help you today?', sender: 'BOT' }
    ]);
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = async (text: string) => {   
        if (!text.trim() || isLoading) return;

        const userMessage: Message = { id: Date.now().toString(), text: text.trim(), sender: 'ME' };
        const updatedMessages = [...messages, userMessage];

        setMessages(updatedMessages);
        setIsLoading(true);

        try {
            const historyForAI = updatedMessages.slice(-6).map(message => ({
                role: message.sender === 'ME' ? 'user' : 'assistant',
                content: message.text
            }));

            const rawBotReply = await sendMessageToBot(historyForAI);
            // Latex sanitizer to make math formulas sent by the bot to be more user friendly
            const cleanReply = rawBotReply
                .replace(/#/g, '')
                .replace(/\*/g, '')
                .replace(/---/g, '')
                .replace(/\\frac\{(.+?)\}\{(.+?)\}/g, '($1) / ($2)') 
                .replace(/\\text\{([^}]+)\}/g, '$1')
                .replace(/\\text/g, '')
                .replace(/\\times/g, ' * ')
                .replace(/\\approx/g, ' ~ ')
                .replace(/\\div/g, ' / ')
                .replace(/\\/g, '') 
                .replace(/[\{\}]/g, '') 
                .replace(/\$/g, '')
                .replace(/[ \t]+/g, ' ') 
                .replace(/\n{3,}/g, '\n\n') 
                .trim();
            
            const botMessage: Message = { id: (Date.now() + 1).toString(), text: cleanReply, sender: 'BOT' };

            setMessages((previous) => [...previous, botMessage]);
        } catch (error) {
            const errorMessage: Message = { id: (Date.now() + 1).toString(), text: 'Sorry, I could not connect to the server.', sender: 'BOT' };

            setMessages((previous) => [...previous, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };
    return { messages, isLoading, sendMessage };
};