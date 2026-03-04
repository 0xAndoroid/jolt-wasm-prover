import { chromium } from 'playwright';

const TIMEOUT = 120_000;

async function run() {
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    page.on('console', msg => {
        process.stderr.write(`[console] ${msg.text()}\n`);
    });

    page.on('pageerror', err => {
        process.stderr.write(`[PAGE ERROR] ${err.message}\n`);
    });

    await page.goto('http://localhost:8080', { waitUntil: 'domcontentloaded' });

    await page.waitForFunction(
        () => document.getElementById('status')?.classList.contains('ready'),
        { timeout: TIMEOUT },
    );
    console.log('WASM ready');

    // Click prove
    await page.click('#page-sha2 .prove-btn');
    console.log('Prove clicked, waiting...');

    // Wait for proof to complete or error
    const proveResult = await page.waitForFunction(
        () => {
            const status = document.getElementById('status');
            const output = document.querySelector('#page-sha2 .output');
            if (status?.classList.contains('error')) return 'ERROR: ' + status.textContent;
            if (output?.textContent?.match(/Proof generated in/)) return 'PROVED';
            return null;
        },
        { timeout: TIMEOUT },
    );
    const proveStatus = await proveResult.jsonValue();
    console.log(`Prove result: ${proveStatus}`);
    if (proveStatus.startsWith('ERROR')) {
        await browser.close();
        console.log(`FAILURE: Proving failed: ${proveStatus}`);
        process.exit(1);
    }

    // Wait for prove button to re-enable
    await page.waitForFunction(
        () => !document.querySelector('#page-sha2 .prove-btn')?.disabled,
        { timeout: TIMEOUT },
    );

    // Click verify
    const verifyBtn = page.locator('#page-sha2 .verify-btn');
    await verifyBtn.waitFor({ state: 'visible', timeout: TIMEOUT });
    await verifyBtn.click();
    console.log('Verify clicked, waiting...');

    // Wait for verification result or error — capture the FULL status text
    const verifyResult = await page.waitForFunction(
        () => {
            const status = document.getElementById('status');
            const statusSpan = status?.querySelector('span') || status;
            const output = document.querySelector('#page-sha2 .output');
            const text = output?.textContent || '';
            if (text.includes('Result: VALID')) return 'VALID';
            if (text.includes('Result: INVALID')) return 'INVALID';
            // Get ALL text from the status area when error
            if (status?.classList.contains('error')) {
                // Traverse all text nodes in status to get full message
                const walker = document.createTreeWalker(status, NodeFilter.SHOW_TEXT);
                let fullText = '';
                while (walker.nextNode()) fullText += walker.currentNode.textContent;
                return 'ERROR: ' + fullText.trim();
            }
            return null;
        },
        { timeout: TIMEOUT },
    );

    const result = await verifyResult.jsonValue();
    console.log(`Verification result: ${result}`);

    // Also grab the full page output for diagnostics
    const outputText = await page.locator('#page-sha2 .output').textContent();
    console.log(`Output log:\n${outputText}`);

    await browser.close();

    if (result === 'VALID') {
        console.log('SUCCESS: Proof verified correctly');
        process.exit(0);
    } else {
        console.log(`FAILURE: ${result}`);
        process.exit(1);
    }
}

run().catch(e => { console.error(e); process.exit(1); });
