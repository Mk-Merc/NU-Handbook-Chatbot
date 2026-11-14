import dns from 'dns';
// Fixes the 'querySrv ECONNREFUSED' error by forcing public DNS, required when using a mobile hotspot
dns.setServers(['8.8.8.8', '8.8.4.4']);

import express from 'express';
import dotenv from 'dotenv';
import chatRoutes from './routes/chatRoutes';

dotenv.config();

const app = express();
app.use(express.json());
app.use(chatRoutes);

const PORT = process.env.PORT;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});