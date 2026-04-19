import random
import re
import os
import json
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'vocabulary.csv')

LEVEL_ORDER = ["初級", "中級", "高級"]
MASTERED_THRESHOLD = 3

# ── 載入單字表（啟動時讀一次）────────────────────────────────────────
try:
    _df = pd.read_csv(CSV_PATH)
    WORD_POOL = _df.to_dict('records')
except Exception as e:
    print(f"[WARNING] Cannot load vocabulary.csv: {e}")
    WORD_POOL = []

# ── 每個等級的單字集合（預先計算）────────────────────────────────────
LEVEL_WORDS = {lv: [w['word'] for w in WORD_POOL if str(w['toeic_target']) == lv]
               for lv in LEVEL_ORDER}
LEVEL_COUNTS = {lv: len(LEVEL_WORDS[lv]) for lv in LEVEL_ORDER}


def _make_blank_pattern(w):
    if w.endswith('y'):
        stem    = re.escape(w[:-1])
        base    = re.escape(w)
        pattern = rf'\b(?:{base}\w*|{stem}(?:ies|ied))\b'
    else:
        base    = re.escape(w)
        pattern = rf'\b{base}\w*\b'
    return re.compile(pattern, re.IGNORECASE)


def _get_question(difficulty, last_words, correct_counts):
    """
    last_words: list，最近 N 題的單字，避免重複
    """
    max_idx  = LEVEL_ORDER.index(difficulty) if difficulty in LEVEL_ORDER else 2
    eligible = [w for w in WORD_POOL
                if str(w['toeic_target']) in LEVEL_ORDER[:max_idx + 1]]
    if not eligible:
        return None

    last_set = set(last_words)
    mastered = {w for w, c in correct_counts.items() if c >= MASTERED_THRESHOLD}
    new_ones = [w for w in eligible if w['word'] not in correct_counts]
    old_ones = [w for w in eligible
                if w['word'] in correct_counts and w['word'] not in mastered]
    all_pool = new_ones + old_ones or eligible

    # 排除最近出現過的題目
    filtered = [w for w in all_pool if w['word'] not in last_set]
    if not filtered:
        filtered = all_pool

    if old_ones and random.random() > 0.7:
        cands      = [w for w in old_ones if w['word'] not in last_set] or old_ones
        target_raw = random.choice(cands)
    else:
        target_raw = random.choice(filtered)

    target = target_raw.copy()

    raw_s = str(target.get('sentence', ""))
    raw_t = str(target.get('trans_s', ""))
    if re.search(r'\d+\.', raw_s):
        s_parts   = [p.strip() for p in re.split(r'\d+\.', raw_s) if p.strip()]
        t_parts   = [p.strip() for p in re.split(r'\d+\.', raw_t) if p.strip()]
        idx       = random.randrange(len(s_parts))
        display_s = s_parts[idx]
        display_t = t_parts[idx] if idx < len(t_parts) else raw_t
    else:
        display_s = raw_s.strip()
        display_t = raw_t.strip()

    final_sentence = display_s
    for w in [w.strip() for w in target['word'].split('/')]:
        base_w = re.sub(r'\(.*?\)', '', w).strip()
        if not base_w:
            continue
        pat = _make_blank_pattern(base_w)
        if pat.search(final_sentence):
            final_sentence = pat.sub("______", final_sentence)
            break

    others      = [w for w in WORD_POOL if w['word'] != target['word']]
    distractors = random.sample(others, min(3, len(others)))
    options     = [{"word": d['word'], "trans": d['translation']} for d in distractors]
    options.append({"word": target['word'], "trans": target['translation']})
    random.shuffle(options)

    return {
        "word":        target['word'],
        "translation": target['translation'],
        "level":       str(target.get('toeic_target', '')),
        "sentence":    final_sentence,
        "trans_s":     display_t,
        "options":     options,
    }


def index(request):
    return render(request, 'toeic_app/index.html', {
        'stats':       json.dumps(LEVEL_COUNTS),
        'level_words': json.dumps(LEVEL_WORDS),
    })


@require_GET
def get_question(request):
    difficulty  = request.GET.get('difficulty', '中級')
    # 接收最近 5 題的歷史，避免重複
    last_raw    = request.GET.get('last_words', '[]')
    counts_raw  = request.GET.get('counts', '{}')
    try:
        last_words = json.loads(last_raw)
    except Exception:
        last_words = []
    try:
        correct_counts = json.loads(counts_raw)
    except Exception:
        correct_counts = {}

    q = _get_question(difficulty, last_words, correct_counts)
    if not q:
        return JsonResponse({"error": "no questions"}, status=404)
    return JsonResponse(q)