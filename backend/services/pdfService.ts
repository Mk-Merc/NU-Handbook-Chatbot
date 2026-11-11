import fs from 'fs';
import { PDFParse } from 'pdf-parse';

const pdfPath = './data/handbook.pdf';

const parsePdf = async () => {
    try {
        // Read the raw binary data from the PDF file
        const dataBuffer = fs.readFileSync(pdfPath);
        const parser = new PDFParse({ data: dataBuffer });
        const result = await parser.getText();
        
        fs.writeFileSync('./data/handbook.txt', result.text);

        console.log(`Extracted ${result.total} pages`);
        
        // Destroy the parser instance to prevent memory leak
        await parser.destroy();
    } catch (err) {
        console.error("Error parsing PDF:", err);
    }
};

parsePdf();