// React-UI gate for W4-UI: drives the actual frontend (not the worker
// directly) on both arms, asserts badge + arm-tagged result line + verify,
// and captures the mandated 375px/1440px screenshots.
// Usage: node gate-ui.mjs <baseUrl> <shotsDir>
import { chromium } from 'playwright';

const BASE = process.argv[2] || 'http://localhost:8102';
const SHOTS = process.argv[3] || '/tmp/w4ui-shots';

const browser = await chromium.launch({
    headless: true,
    channel: process.env.PW_CHANNEL || 'chrome',
});

async function uiRun({ label, url, viewport, prove }) {
    const page = await browser
        .newContext({ viewport })
        .then((c) => c.newPage());
    page.on('console', (m) => process.stderr.write(`[${label}] ${m.text()}\n`));
    await page.goto(url, { waitUntil: 'domcontentloaded' });

    await page.locator('#status', { hasText: /Ready|Error/ }).waitFor({ timeout: 60000 });
    const status = await page.locator('#status').innerText();
    const badge = await page.locator('#gpu-badge').first().innerText();

    let proveLine = null;
    let verifyBadge = null;
    let logText = '';
    if (prove) {
        await page.locator('.prove-btn').first().click();
        const done = page.locator('#page-sha2 pre, #page-sha2 [class*=output], #page-sha2');
        await page
            .getByText(/Proof generated in [\d.]+s \((GPU|CPU)\)/)
            .waitFor({ timeout: 120000 });
        logText = await page.locator('#page-sha2').innerText();
        proveLine = logText.match(/Proof generated in [\d.]+s \((GPU|CPU)\)/)?.[0] ?? null;

        await page.locator('.verify-btn').first().click();
        await page.getByText('Result: VALID').waitFor({ timeout: 60000 });
        verifyBadge = (await page.locator('#page-sha2').innerText()).includes('Valid');
        logText = await page.locator('#page-sha2').innerText();
    }

    const overflow = await page.evaluate(() => ({
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
    }));

    await page.screenshot({ path: `${SHOTS}/${label}.png`, fullPage: false });
    await page.context().close();
    return {
        label,
        status: status.trim(),
        badge: badge.trim(),
        proveLine,
        verified: verifyBadge,
        proofSizeLine: logText.match(/Proof size: [\d.]+ KB/)?.[0] ?? null,
        noCompressedLine: !logText.includes('compressed'),
        overflow,
        overflowOk: overflow.scrollWidth <= overflow.clientWidth,
    };
}

const results = [];
results.push(await uiRun({
    label: 'on-1440', url: BASE, viewport: { width: 1440, height: 900 }, prove: true,
}));
results.push(await uiRun({
    label: 'off-375', url: `${BASE}/?webgpu=0`, viewport: { width: 375, height: 812 }, prove: true,
}));
results.push(await uiRun({
    label: 'on-375', url: BASE, viewport: { width: 375, height: 812 }, prove: false,
}));
results.push(await uiRun({
    label: 'off-1440', url: `${BASE}/?webgpu=0`, viewport: { width: 1440, height: 900 }, prove: false,
}));

await browser.close();
console.log(JSON.stringify(results, null, 2));
const pass =
    results[0].badge === 'GPU' && results[0].proveLine?.includes('(GPU)') && results[0].verified &&
    results[1].badge === 'CPU' && results[1].proveLine?.includes('(CPU)') && results[1].verified &&
    results[2].badge === 'GPU' && results[3].badge === 'CPU' &&
    results.every((r) => r.overflowOk) &&
    results.filter((r) => r.proveLine).every((r) => r.proofSizeLine && r.noCompressedLine);
console.log(pass ? 'UI-GATE PASS' : 'UI-GATE FAIL');
process.exit(pass ? 0 : 1);
