import { ThemeProvider as ArloThemeProvider } from '@/lib/arloui/theme-provider';
import { useArloFonts } from '@/lib/arloui/fonts';
import { useChatbot } from './hooks/useChatbot';
import { ChatScreen } from './pages/ChatScreen';

export default function App() {
    const [fontsLoaded] = useArloFonts();

    const { 
        messages, 
        isLoading, 
        sendMessage 
    } = useChatbot();

    if (!fontsLoaded) return null;

    return (
        <ArloThemeProvider>
            <ChatScreen
                messages={messages}
                isLoading={isLoading}
                onSendMessage={sendMessage}
            />
        </ArloThemeProvider>
    );
}