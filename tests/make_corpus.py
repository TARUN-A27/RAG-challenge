#!/usr/bin/env python3
"""Generate a bigger, nastier corpus than the sample one, with known answers.

Same conventions as the graded corpus (every file type; withdrawn revisions; an encrypted PDF, an
unreadable file, an unknown type, empty directories) but ~70 files and look-alikes everywhere:
near-identical constant names across services, memos that repeat a value without being its source,
a ticket reachable only through a log line, and prices that exist only inside the encrypted file.

    .venv/bin/python tests/make_corpus.py        # -> tests/extra/{corpus/, questions.json, ocr_fixtures.json}

Seeded, so every run produces the same corpus. Needs fpdf2 (dev only).
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import docx
import openpyxl
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter

OUT = Path(__file__).resolve().parent / "extra"
C = OUT / "corpus"
R = random.Random(7)

PRODUCTS = [("VX-12", "Falcon"), ("VX-18", "Osprey"), ("KR-7", "Heron"), ("KR-9", "Ibis"), ("ZN-30", "Condor"),
            ("ZN-31", "Kestrel"), ("PL-4", "Merlin"), ("PL-8", "Harrier"), ("QS-2", "Swift"), ("QS-5", "Tern")]
Q = []              # questions
FIX = {}            # image path -> what a good vision model would transcribe


def put(rel, text="", mode="w"):
    p = C / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if text is not None:
        (p.write_bytes if mode == "wb" else p.write_text)(text)
    return p


def pdf(rel, lines):
    p = put(rel, None)
    d = FPDF()
    d.add_page()
    d.set_font("Helvetica", size=11)
    for ln in lines:
        d.cell(0, 6, ln, new_x="LMARGIN", new_y="NEXT")
    d.output(str(p))
    return p


def ask(q, answer, cites, **kw):
    Q.append({"n": len(Q) + 1, "query": q, "expected_answer": answer, "answer_aliases": kw.pop("aliases", []),
              "expected_citations": cites, **kw})


font = lambda n: ImageFont.load_default(size=n)     # noqa: E731


def label_png(rel, pid, rev, serial):
    im = Image.new("RGB", (760, 440), (232, 232, 226))
    d = ImageDraw.Draw(im)
    d.rectangle((50, 45, 710, 385), fill="white", outline="black", width=4)
    for y, t in ((70, "Halcyon Labs"), (130, f"MODEL   {pid}"), (190, f"BOARD REVISION   {rev}"), (250, f"SN  {serial}")):
        d.text((85, y), t, fill="black", font=font(34))
    d.text((85, 320), "MADE IN VIETNAM", fill="black", font=font(24))
    im.rotate(-3, expand=True, fillcolor=(232, 232, 226)).save(put(rel, None))
    FIX[f"{rel}"] = f"Halcyon Labs\nMODEL {pid}\nBOARD REVISION {rev}\nSN {serial}\nMADE IN VIETNAM"


def pinout_png(rel, pid, pins, hot):
    im = Image.new("RGB", (1000, 560), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), f"{pid} backplane connector - pin assignment", fill="black", font=font(30))
    d.rectangle((40, 90, 960, 300), outline="black", width=4)
    x = 60
    for pin, sig in pins:
        d.rectangle((x, 120, x + 118, 190), outline="black", width=2)
        d.text((x + 59, 155), pin, fill="black", font=font(28), anchor="mm")
        d.line((x + 59, 190, x + 59, 212), fill="black", width=2)
        d.text((x + 59, 228), sig, fill="black", font=font(18), anchor="mm")
        x += 150
    d.text((40, 340), f"{hot} is asserted low when the die exceeds the warning threshold.", fill="black", font=font(26))
    im.save(put(rel, None))
    FIX[rel] = (f"{pid} backplane connector - pin assignment\n" + "\n".join(f"{p}: {s}" for p, s in pins)
                + f"\n{hot} is asserted low when the die exceeds the warning threshold.")


def main():
    if OUT.exists():
        for dp, dn, fn in os.walk(OUT):             # the unreadable file / locked dirs from a previous run
            for n in dn + fn:
                try:
                    os.chmod(Path(dp, n), 0o755)
                except OSError:
                    pass
        shutil.rmtree(OUT)
    C.mkdir(parents=True)

    # ---- products -----------------------------------------------------------------------------------
    facts = {}
    for i, (pid, name) in enumerate(PRODUCTS):
        f = facts[pid] = dict(
            name=name, tj=R.randint(78, 104), power=R.randrange(180, 620, 5), mem=R.choice([24, 32, 48, 64, 96]),
            fan=f"HLX-FAN-{R.randint(1000, 4999)}-{R.choice('ABCD')}", legacy=f"HLX-FAN-{R.randint(5000, 9999)}-A",
            lead=R.randint(7, 60), price=R.randint(900, 9000), rev=f"REV-{R.choice('ABCDEF')}{R.randint(1, 4)}",
            quarter=f"Q{R.randint(1, 4)} FY{R.choice([27, 28])}")
        f["tj_old"] = f["tj"] + R.randint(4, 14)
        f["mem_old"] = f["mem"] - 8

    # ---- datasheets: current revision for all, a withdrawn one for six -------------------------------
    for i, (pid, name) in enumerate(PRODUCTS):
        f = facts[pid]
        pdf(f"specs/{pid.lower()}_datasheet_r2.pdf", [
            f'Halcyon Labs {pid} "{name}" Accelerator', "Datasheet, revision 2 - supersedes revision 1",
            "Electrical and thermal", f"Maximum junction temperature .......... {f['tj']} C",
            f"Board power (TBP) ..................... {f['power']} W", "Memory",
            f"Capacity .............................. {f['mem']} GiB HBM3e", "Cooling",
            f"Fan assembly part ..................... {f['fan']}", "Revision history",
            f"r2  Junction temperature corrected to {f['tj']} C after production characterisation. Revision 1 is withdrawn.",
            "r1  Initial release, preliminary silicon."])
        if i < 6:
            tag = "WITHDRAWN" if i % 3 else "SUPERSEDED"
            pdf(f"specs/{pid.lower()}_datasheet_r1_{tag}.pdf", [
                f'Halcyon Labs {pid} "{name}" Accelerator', f"Datasheet, revision 1 - {tag}, REPLACED BY REVISION 2",
                "Preliminary silicon. Do not design to these numbers.", f"Maximum junction temperature .......... {f['tj_old']} C",
                f"Capacity .............................. {f['mem_old']} GiB HBM3e"])

    # ---- roadmap documents: sampling quarters, in a table --------------------------------------------
    for fy in (27, 28):
        d = docx.Document()
        d.add_heading(f"Halcyon Labs - FY{fy} Accelerator Roadmap (Internal)", 1)
        d.add_paragraph("Dates are targets, not commitments.")
        t = d.add_table(rows=1, cols=3)
        t.rows[0].cells[0].text, t.rows[0].cells[1].text, t.rows[0].cells[2].text = "product", "milestone", "target"
        for pid, name in PRODUCTS:
            if facts[pid]["quarter"].endswith(str(fy)):
                row = t.add_row().cells
                row[0].text, row[1].text, row[2].text = f"{pid} {name}", "customer sampling", facts[pid]["quarter"]
                row = t.add_row().cells
                row[0].text, row[1].text, row[2].text = f"{pid} {name}", "volume production", f"Q{R.randint(1, 4)} FY{fy + 1}"
        d.save(put(f"planning/roadmap_fy{fy}.docx", None))

    # ---- RMA parts workbook ---------------------------------------------------------------------------
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Fans"
    ws.append(["part_number", "description", "compatible_with", "unit_cost_usd", "lead_time_days"])
    for pid, _ in PRODUCTS:
        f = facts[pid]
        ws.append([f["fan"], "Fan assembly, dual-rotor, field replaceable", pid, round(R.uniform(40, 120), 2), f["lead"]])
        ws.append([f["legacy"], "Fan assembly, single-rotor (legacy)", f"{pid[:2]}-{int(pid[3:]) - 1}", round(R.uniform(30, 70), 2), R.randint(7, 40)])
    ws = wb.create_sheet("Heatsinks")
    ws.append(["part_number", "description", "compatible_with", "unit_cost_usd", "lead_time_days"])
    for pid, _ in PRODUCTS:
        ws.append([f"HLX-HSK-{R.randint(1000, 9999)}-A", "Heatsink, vapour chamber", pid, R.randint(150, 300), R.randint(14, 60)])
    wb.create_sheet("Notes").append(["Lead times are supplier-quoted and exclude customs."])
    wb.save(put("support/rma_parts.xlsx", None))

    # ---- bug databases, release notes, logs -----------------------------------------------------------
    versions = ["5.1.0", "5.1.1", "5.1.2", "5.2.0", "5.2.1", "5.2.2", "5.3.0"]
    summaries = ["Sensor reads high under sustained small-batch load", "Doorbell hang after link retrain",
                 "Occupancy counter drifts after long uptime", "Fan curve too aggressive at idle",
                 "ECC scrub interval ignored", "Clock reduced longer than necessary after thermal event",
                 "Telemetry export drops the last sample", "Firmware update leaves stale NVRAM entry"]
    tickets, n = {}, 1000
    for year in (2025, 2026):
        rows = []
        for _ in range(40):
            n += R.randint(1, 9)
            fixed = R.choice(versions + [""])
            tickets[f"HLX-{n}"] = (year, fixed)
            rows.append([f"HLX-{n}", R.choice(["power", "pcie", "scheduler", "thermal", "firmware"]), R.choice(["S2", "S3", "S4"]),
                         R.choice(summaries), "closed" if fixed else "open", fixed])
        put(f"support/bugs_{year}.csv", "ticket,component,severity,summary,status,fixed_in\n"
            + "\n".join(",".join(f'"{c}"' if "," in c else c for c in r) for r in rows) + "\n")
    for v in versions[:4]:
        put(f"engineering/release_notes_{v}.txt", f"Halcyon firmware {v} - release notes\n\nScheduler and thermal tuning. "
            f"Fixes are tracked per ticket in the bug database, which is authoritative.\n")

    err = [("E5127", "thermal throttle engaged on die 0, clocks reduced to 60%"), ("E6310", "ECC uncorrectable error on HBM stack 2"),
           ("E7044", "PCIe link retrain failed, downgraded to x8"), ("E4419", "fan tach below threshold on fan 1")]
    fixed_tickets = [t for t, (_, v) in tickets.items() if v]
    log_files = []
    for k in range(8):
        node, date = R.randint(1, 12), f"2026-08-{R.randint(10, 28):02d}"
        rel = f"logs/node-{node}_{date}.log"
        if rel in log_files:
            continue
        log_files.append(rel)
        code, msg = R.choice(err)
        tk = R.choice(fixed_tickets)
        lines = [f"{date}T03:{m:02d}:{R.randint(0, 59):02d}Z node-{node} halcyon[{R.randint(1000, 9999)}]: INFO  scheduler: batch {R.randint(10000, 99999)} accepted"
                 for m in range(8, 24)]
        lines.insert(10, f"{date}T03:18:41Z node-{node} halcyon[2211]: ERROR {code}: {msg}")
        lines.append(f"{date}T03:30:03Z node-{node} halcyon[2211]: INFO  incident logged against {tk}")
        put(rel, "\n".join(lines) + "\n")
        facts[rel] = dict(node=node, date=date, code=code, ticket=tk, year=tickets[tk][0], fixed=tickets[tk][1], msg=msg)

    # ---- services: near-identical constants ------------------------------------------------------------
    svc = {}
    for name in ["ingest", "export", "scheduler", "billing", "audit", "notify"]:
        to, old, mr = R.choice([30, 45, 90, 120, 150, 240, 300, 600]), R.choice([15, 20, 60]), R.randint(2, 9)
        svc[name] = (to, mr)
        put(f"engineering/svc_{name}.py", f'"""{name.capitalize()} service."""\n\n# Seconds a batch may wait before it is abandoned.\n'
            f"# Raised from {old} after the 2026-08 backlog incident.\nDEFAULT_BATCH_TIMEOUT_S = {to}\n\nMAX_RETRIES = {mr}\n"
            f'STAGING_ROOT = "/var/lib/halcyon/{name}"\n\n\ndef accept(batch, timeout_s=DEFAULT_BATCH_TIMEOUT_S):\n    return batch.enqueue(timeout_s)\n')

    # ---- memos: repeat a value without being its source ------------------------------------------------
    for pid in ("VX-12", "KR-9", "PL-4"):
        f = facts[pid]
        put(f"planning/memo_{pid.lower()}_thermal.txt", f"Thermal review notes\n\nWe discussed the {pid} cooling budget today. "
            f"The Tj limit of {f['tj']} C is already in the datasheet; nobody proposed changing it.\n")
    put("planning/memo_offsite.txt", "Offsite notes\n\nCatering, travel, and the usual complaints about the parking garage.\n")

    # ---- pictures ----------------------------------------------------------------------------------------
    for pid in ("VX-12", "ZN-30", "QS-5"):
        label_png(f"support/label_{pid.lower()}.png", pid, facts[pid]["rev"], f"HL{R.randint(100, 999)}-{R.randint(100000, 999999)}")
    pins = {}
    for pid, hot in (("KR-7", "THERM_ALERT#"), ("PL-8", "FAN_FAIL#")):
        names = ["GND", "SDA", "SCL", hot, "PRSNT#", "GND"]
        R.shuffle(names)
        base = R.randint(2, 6) * 10
        pins[pid] = (hot, [(f"C{base + j}", s) for j, s in enumerate(names)])
        pinout_png(f"specs/{pid.lower()}_pinout.png", pid, pins[pid][1], hot)

    # ---- the four things the folder does to you --------------------------------------------------------
    (C / "archive" / "2024").mkdir(parents=True)
    put("vendor/telemetry_capture.dat", bytes(range(256)) * 16, "wb")
    plain = pdf("vendor/_plain.pdf", ["Supplier agreement", "Unit price at 10,000 unit volume:"]
                + [f"{pid} .......... {facts[pid]['price']} USD" for pid, _ in PRODUCTS])
    w = PdfWriter(clone_from=PdfReader(plain))
    w.encrypt("hunter2", algorithm="RC4-128")
    with open(C / "vendor/supplier_agreement_ENCRYPTED.pdf", "wb") as fh:
        w.write(fh)
    plain.unlink()
    put("vendor/internal_audit.txt", "The 2025 internal audit was signed by M. Okafor.\n")
    os.chmod(C / "vendor/internal_audit.txt", 0)

    # ---- questions ------------------------------------------------------------------------------------
    f = facts["VX-12"]
    ask("What is the maximum junction temperature of the VX-12?", str(f["tj"]), ["specs/vx-12_datasheet_r2.pdf"], note="withdrawn r1 + a memo repeating the value")
    f = facts["ZN-31"]
    ask("What is the maximum junction temperature of the ZN-31?", str(f["tj"]), ["specs/zn-31_datasheet_r2.pdf"])
    f = facts["PL-4"]
    ask(f"What is the board power of the {f['name']}?", str(f["power"]), ["specs/pl-4_datasheet_r2.pdf"], note="asked by codename")
    f = facts["KR-9"]
    rd = "planning/roadmap_fy27.docx" if f["quarter"].endswith("27") else "planning/roadmap_fy28.docx"
    ask(f"In which quarter does the {f['name']} enter customer sampling?", f["quarter"], [rd], aliases=[f["quarter"].replace(" ", "")], note="docx table")
    f = facts["QS-2"]
    ask("What is the part number of the field-replaceable fan assembly for the QS-2?", f["fan"], ["support/rma_parts.xlsx"], note="legacy single-rotor fan is the near miss")
    f = facts["VX-18"]
    ask(f"What is the lead time, in days, of part {f['fan']}?", str(f["lead"]), ["support/rma_parts.xlsx"])
    t, (yr, fx) = next(iter((t, v) for t, v in tickets.items() if v[1]))
    ask(f"Which firmware version fixed ticket {t}?", fx, [f"support/bugs_{yr}.csv"])
    lg = facts[log_files[0]]
    ask(f"What error code is logged on node-{lg['node']} on {lg['date']}?", lg["code"], [log_files[0]])
    ask("What is the default batch timeout, in seconds, in the export service?", str(svc["export"][0]), ["engineering/svc_export.py"], note="a comment names an old value")
    ask("What is the maximum number of retries in the billing service?", str(svc["billing"][1]), ["engineering/svc_billing.py"])
    lg = facts[log_files[1]]
    ask(f"The production log for node-{lg['node']} on {lg['date']} shows an incident. Which firmware release fixed the underlying defect?",
        lg["fixed"], [log_files[1], f"support/bugs_{lg['year']}.csv"], note="two files: the log names the ticket")
    f = facts["ZN-30"]
    ask("What is the lead time in days of the fan assembly specified in the ZN-30 datasheet?", str(f["lead"]),
        ["specs/zn-30_datasheet_r2.pdf", "support/rma_parts.xlsx"], note="two files: the datasheet names the part")
    ask("What board revision is printed on the VX-12 asset label?", facts["VX-12"]["rev"], ["support/label_vx-12.png"], note="picture only")
    ask("What board revision is printed on the QS-5 asset label?", facts["QS-5"]["rev"], ["support/label_qs-5.png"], note="picture only")
    hot, pl = pins["KR-7"]
    ask(f"Which backplane pin carries {hot} on the KR-7?", next(p for p, s in pl if s == hot), ["specs/kr-7_pinout.png"], note="picture only")
    hot, pl = pins["PL-8"]
    ask(f"Which backplane pin carries {hot} on the PL-8?", next(p for p, s in pl if s == hot), ["specs/pl-8_pinout.png"], note="picture only")
    ask("What is the unit price of the VX-18 at 10,000 unit volume?", "", [], unanswerable=True, note="exists only in the encrypted PDF")
    ask("What is the maximum junction temperature of the XQ-99?", "", [], unanswerable=True, note="no such product")
    ask("Who signed the 2025 internal audit?", "", [], unanswerable=True, note="exists only in the unreadable file")

    (OUT / "questions.json").write_text(json.dumps({"corpus": "corpus", "queries": Q}, indent=1))
    (OUT / "ocr_fixtures.json").write_text(json.dumps(FIX, indent=1))
    print(f"{sum(len(fn) for _, _, fn in os.walk(C))} files, {len(Q)} questions -> {OUT}")


if __name__ == "__main__":
    main()
