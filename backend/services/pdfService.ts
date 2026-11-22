import fs from 'fs';
import { PDFParse } from 'pdf-parse';

const PDF_PATH = './data/handbook.pdf';
const TEXT_PATH = './data/handbook.txt';

const parsePdf = async () => {
    try {
        // Read the raw binary data from the PDF file
        const dataBuffer = fs.readFileSync(PDF_PATH);
        const parser = new PDFParse({ data: dataBuffer });
        const result = await parser.getText();
        
        fs.writeFileSync(TEXT_PATH, result.text);

        console.log(`Extracted ${result.total} pages`);
        
        // Destroy the parser instance to prevent memory leak
        await parser.destroy();
    } catch (error) {
        console.error("Error parsing PDF:", error);
    }
};

// Run if this file is executed directly
if (require.main === module) {
    parsePdf();
}