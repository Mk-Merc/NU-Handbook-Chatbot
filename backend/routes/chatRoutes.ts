import { Router } from 'express';
import { handleChat } from '../controllers/chatController';

const router = Router();

router.post('/api/chat', handleChat);

export default router;