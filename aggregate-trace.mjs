// W2c: aggregate SELF-time from a wasm_tracing Chrome Trace Format dump.
// Events carry real per-thread tids (see src/wasm_tracing.rs): B/E are
// matched per (tid,name) LIFO, containment forests are built per tid, so
// nesting is exact. The driver tid is the one holding the `prove` span;
// its prove_stageN intervals define the stage timeline. Spans on other
// (rayon worker) tids are CPU time, attributed to the stage whose driver
// interval contains their start.
//
// Usage: node aggregate-trace.mjs <trace.json> [more.json ...]

import { readFileSync } from 'fs';

function buildForest(intervals) {
    // intervals: sorted by (start asc, end desc). Returns per-interval
    // childDur + partial-overlap stats (should be ~0 with real tids).
    let partialOverlaps = 0, partialOverlapDur = 0;
    const stack = [];
    for (const cur of intervals) {
        while (stack.length && stack[stack.length - 1].stackEnd <= cur.start) stack.pop();
        const parent = stack.length ? stack[stack.length - 1] : null;
        cur.stackEnd = cur.end;
        if (parent) {
            if (cur.end > parent.stackEnd + 1) {
                partialOverlaps++;
                partialOverlapDur += cur.end - parent.stackEnd;
                cur.stackEnd = parent.stackEnd;
            }
            parent.childDur = (parent.childDur || 0) + (cur.stackEnd - cur.start);
            cur.parent = parent;
        }
        stack.push(cur);
    }
    return { partialOverlaps, partialOverlapDur };
}

function analyze(file) {
    const events = JSON.parse(readFileSync(file, 'utf-8'));
    const open = new Map(); // `${tid}:${name}` -> stack of ts
    const byTid = new Map(); // tid -> intervals
    let unmatchedE = 0;
    for (const e of events) {
        if (e.ph !== 'B' && e.ph !== 'E') continue;
        const key = `${e.tid}:${e.name}`;
        if (e.ph === 'B') {
            if (!open.has(key)) open.set(key, []);
            open.get(key).push(e.ts);
        } else {
            const st = open.get(key);
            if (st && st.length) {
                if (!byTid.has(e.tid)) byTid.set(e.tid, []);
                byTid.get(e.tid).push({ name: e.name, start: st.pop(), end: e.ts, tid: e.tid });
            } else unmatchedE++;
        }
    }
    let unclosedB = 0;
    for (const st of open.values()) unclosedB += st.length;

    let qc = { unmatchedE, unclosedB, partialOverlaps: 0, partialOverlapDur: 0, intervals: 0 };
    for (const ivs of byTid.values()) {
        ivs.sort((a, b) => a.start - b.start || b.end - a.end);
        const r = buildForest(ivs);
        qc.partialOverlaps += r.partialOverlaps;
        qc.partialOverlapDur += r.partialOverlapDur;
        qc.intervals += ivs.length;
    }

    // Driver tid = the one holding the longest `prove` interval.
    let driverTid = null, proveIv = null;
    for (const [tid, ivs] of byTid) {
        for (const iv of ivs) {
            if (iv.name === 'prove' && (!proveIv || iv.end - iv.start > proveIv.end - proveIv.start)) {
                proveIv = iv;
                driverTid = tid;
            }
        }
    }
    if (!proveIv) throw new Error(`${file}: no 'prove' span found`);

    const driverIvs = byTid.get(driverTid);
    const stageIvs = driverIvs
        .filter((iv) => iv.name.startsWith('prove_stage'))
        .sort((a, b) => a.start - b.start);
    const stageOf = (ts) => {
        for (const st of stageIvs) if (ts >= st.start && ts < st.end) return st.name;
        return '(outside stages)';
    };

    // Driver self-time (wall attribution) and worker CPU per name, per stage.
    const driverSelf = new Map(); // name -> {self, count}
    const workerCpu = new Map(); // name -> {self, count}
    const stageDriverSelf = new Map(); // stage -> Map(name -> self)
    const stageWorkerCpu = new Map(); // stage -> cpu us total
    const stageWorkerByName = new Map(); // stage -> Map(name -> cpu us)
    const add = (map, key, v) => map.set(key, (map.get(key) || 0) + v);

    for (const [tid, ivs] of byTid) {
        const isDriver = tid === driverTid;
        for (const iv of ivs) {
            const dur = iv.end - iv.start;
            const self = Math.max(0, dur - (iv.childDur || 0));
            const stage = iv.name.startsWith('prove_stage') ? iv.name : stageOf(iv.start);
            const agg = isDriver ? driverSelf : workerCpu;
            const rec = agg.get(iv.name) || { self: 0, count: 0 };
            rec.self += self;
            rec.count++;
            agg.set(iv.name, rec);
            if (isDriver) {
                if (!stageDriverSelf.has(stage)) stageDriverSelf.set(stage, new Map());
                add(stageDriverSelf.get(stage), iv.name, self);
            } else {
                add(stageWorkerCpu, stage, self);
                if (!stageWorkerByName.has(stage)) stageWorkerByName.set(stage, new Map());
                add(stageWorkerByName.get(stage), iv.name, self);
            }
        }
    }

    const stageWalls = new Map();
    for (const st of stageIvs) add(stageWalls, st.name, st.end - st.start);

    const engineIv = driverIvs.find((iv) => iv.name === 'engine::prove');

    return {
        file, qc, tids: byTid.size, driverTid,
        proveWall: proveIv.end - proveIv.start,
        engineWall: engineIv ? engineIv.end - engineIv.start : null,
        stageWalls, driverSelf, workerCpu, stageDriverSelf, stageWorkerCpu, stageWorkerByName,
    };
}

const ms = (us) => (us / 1000).toFixed(1);
const s = (us) => (us / 1e6).toFixed(2);

for (const file of process.argv.slice(2)) {
    const a = analyze(file);
    console.log(`\n===== ${a.file} =====`);
    console.log(
        `tids=${a.tids} driver=${a.driverTid} intervals=${a.qc.intervals} ` +
        `unmatchedE=${a.qc.unmatchedE} unclosedB=${a.qc.unclosedB} ` +
        `partialOverlaps=${a.qc.partialOverlaps} (${ms(a.qc.partialOverlapDur)} ms)`,
    );
    console.log(`prove wall: ${s(a.proveWall)} s   engine::prove wall: ${s(a.engineWall)} s`);

    console.log('\n-- stage walls (driver) --');
    const stages = [...a.stageWalls.entries()].sort();
    let stageSum = 0;
    for (const [st, dur] of stages) {
        stageSum += dur;
        const cpu = a.stageWorkerCpu.get(st) || 0;
        console.log(
            `${st.padEnd(16)} ${s(dur).padStart(8)} s  ${(dur / a.proveWall * 100).toFixed(1).padStart(5)}%` +
            `   workerCPU ${s(cpu).padStart(8)} s (${(cpu / dur).toFixed(1)}x wall)`,
        );
    }
    console.log(`${'(sum stages)'.padEnd(16)} ${s(stageSum).padStart(8)} s  ${(stageSum / a.proveWall * 100).toFixed(1).padStart(5)}%`);

    console.log('\n-- top-30 driver self-time (wall) --');
    for (const [name, r] of [...a.driverSelf.entries()].sort((x, y) => y[1].self - x[1].self).slice(0, 30)) {
        console.log(
            `${ms(r.self).padStart(10)} ms  ${(r.self / a.proveWall * 100).toFixed(1).padStart(5)}%  ` +
            `n=${String(r.count).padStart(5)}  ${name}`,
        );
    }

    console.log('\n-- top-30 worker CPU (sum across threads) --');
    for (const [name, r] of [...a.workerCpu.entries()].sort((x, y) => y[1].self - x[1].self).slice(0, 30)) {
        console.log(`${ms(r.self).padStart(10)} ms  n=${String(r.count).padStart(5)}  ${name}`);
    }

    console.log('\n-- per-stage rollup (driver-self >=2% of stage wall; worker top-5) --');
    for (const [st, wall] of stages) {
        console.log(`\n${st} (wall ${s(wall)} s):`);
        const dm = a.stageDriverSelf.get(st) || new Map();
        for (const [name, self] of [...dm.entries()].sort((x, y) => y[1] - x[1])) {
            if (self < wall * 0.02) continue;
            console.log(`  drv ${ms(self).padStart(10)} ms  ${(self / wall * 100).toFixed(1).padStart(5)}%  ${name}`);
        }
        const wm = a.stageWorkerByName.get(st) || new Map();
        for (const [name, cpu] of [...wm.entries()].sort((x, y) => y[1] - x[1]).slice(0, 5)) {
            console.log(`  cpu ${ms(cpu).padStart(10)} ms  ${name}`);
        }
    }
}
