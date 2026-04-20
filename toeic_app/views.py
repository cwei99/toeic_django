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

LEVEL_ORDER        = ["初級", "中級", "高級"]
MASTERED_THRESHOLD = 3
NORMAL_GAP         = 10   # 一般題目至少間隔幾題
MASTERED_GAP       = 15   # 精熟題至少間隔幾題

# ── 載入 NLP 工具 ────────────────────────────────────────────────
try:
    import spacy
    from lemminflect import getInflection, getLemma
    _nlp = spacy.load("en_core_web_sm")
    NLP_READY = True
except Exception as e:
    print(f"[WARNING] NLP not available: {e}")
    NLP_READY = False

# ── 載入單字表 ───────────────────────────────────────────────────
try:
    _df       = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    WORD_POOL = _df.to_dict('records')
except Exception as e:
    print(f"[WARNING] Cannot load vocabulary.csv: {e}")
    WORD_POOL = []

LEVEL_WORDS  = {lv: [w['word'] for w in WORD_POOL if str(w['toeic_target']) == lv]
                for lv in LEVEL_ORDER}
LEVEL_COUNTS = {lv: len(LEVEL_WORDS[lv]) for lv in LEVEL_ORDER}
VERB_POOL    = [w for w in WORD_POOL if 'v.' in str(w.get('translation', ''))]

_INFLECT_TAG = {'VBD', 'VBZ', 'VBP', 'VBG', 'VBN', 'VB'}

# ── 不規則名詞複數對照表 ──────────────────────────────────────────
_IRREGULAR_NOUNS = {
    'shelves': 'shelf', 'knives': 'knife', 'leaves': 'leaf',
    'loaves': 'loaf', 'wolves': 'wolf', 'halves': 'half',
    'scarves': 'scarf', 'lives': 'life', 'wives': 'wife',
    'thieves': 'thief', 'feet': 'foot', 'teeth': 'tooth',
    'men': 'man', 'women': 'woman', 'children': 'child',
    'mice': 'mouse', 'geese': 'goose', 'oxen': 'ox',
    'people': 'person', 'criteria': 'criterion', 'phenomena': 'phenomenon',
    'data': 'datum', 'media': 'medium', 'analyses': 'analysis',
    'bases': 'basis', 'crises': 'crisis', 'theses': 'thesis',
}
# 反查：原形 → 複數
_NOUN_PLURAL = {v: k for k, v in _IRREGULAR_NOUNS.items()}

# ── 預建「所有變化形 → (原形, tag)」對照表 ────────────────────────
_SURFACE_TO_BASE: dict = {}

if NLP_READY:
    for _entry in VERB_POOL:
        for _part in _entry['word'].split('/'):
            _base = re.sub(r'\(.*?\)', '', _part).strip().lower()
            if not _base:
                continue
            for _tag in ['VB', 'VBD', 'VBZ', 'VBP', 'VBG', 'VBN']:
                _forms = getInflection(_base, tag=_tag)
                if _forms:
                    for _f in _forms:
                        _key = _f.lower()
                        if _key not in _SURFACE_TO_BASE:
                            _SURFACE_TO_BASE[_key] = (_base, _tag)


def _inflect_word(word, tag):
    if not NLP_READY or not tag:
        return re.sub(r'\(.*?\)', '', word.split('/')[0]).strip()
    base   = re.sub(r'\(.*?\)', '', word.split('/')[0]).strip()
    result = getInflection(base, tag=tag)
    return result[0] if result else base


def _make_blank_pattern(w):
    if w.endswith('y'):
        stem    = re.escape(w[:-1])
        base    = re.escape(w)
        pattern = rf'\b(?:{base}\w*|{stem}(?:ies|ied))\b'
    else:
        base    = re.escape(w)
        pattern = rf'\b{base}\w*\b'
    return re.compile(pattern, re.IGNORECASE)


def _find_and_blank(sentence, target_word, is_verb):
    base_forms = set()
    for part in target_word.split('/'):
        base = re.sub(r'\(.*?\)', '', part).strip().lower()
        if base:
            base_forms.add(base)

    # 先把句子裡的 word1/word2 斜線變體合併成第一個詞
    parts = [re.sub(r'\(.*?\)', '', p).strip() for p in target_word.split('/') if p.strip()]
    if len(parts) >= 2:
        for i, a in enumerate(parts):
            for b in parts[i+1:]:
                slash_pat = re.compile(
                    rf'\b{re.escape(a)}/{re.escape(b)}\b|\b{re.escape(b)}/{re.escape(a)}\b',
                    re.IGNORECASE
                )
                sentence = slash_pat.sub(a, sentence)

    # 方法一：預建對照表（處理 stuck/went 等不規則動詞變化）
    if NLP_READY and is_verb:
        doc = _nlp(sentence)
        for token in doc:
            surface = token.text.lower()
            if surface in _SURFACE_TO_BASE:
                orig_base, tag = _SURFACE_TO_BASE[surface]
                if orig_base in base_forms:
                    blanked = sentence[:token.idx] + '______' + sentence[token.idx + len(token.text):]
                    return blanked, tag
            # 備用：spaCy lemma
            if token.lemma_.lower() in base_forms and token.tag_ in _INFLECT_TAG:
                blanked = sentence[:token.idx] + '______' + sentence[token.idx + len(token.text):]
                return blanked, token.tag_

    # 方法二：不規則名詞複數對照表（shelves→shelf 等）
    if not is_verb:
        for base_f in base_forms:
            plural = _NOUN_PLURAL.get(base_f)  # shelf → shelves
            targets = [base_f]
            if plural:
                targets.append(plural)
            for t in targets:
                pat = re.compile(rf'\b{re.escape(t)}\b', re.IGNORECASE)
                if pat.search(sentence):
                    return pat.sub('______', sentence), None

    # 方法三：regex fallback（規則變化）
    for part in target_word.split('/'):
        base_w = re.sub(r'\(.*?\)', '', part).strip()
        if not base_w:
            continue
        pat = _make_blank_pattern(base_w)
        m   = pat.search(sentence)
        if m:
            return pat.sub('______', sentence), None

    return sentence, None


def _get_question(difficulty, history, correct_counts):
    """
    history: list，最近出現過的單字（index 0 = 最新），用來計算間隔
    """
    max_idx  = LEVEL_ORDER.index(difficulty) if difficulty in LEVEL_ORDER else 2
    eligible = [w for w in WORD_POOL
                if str(w['toeic_target']) in LEVEL_ORDER[:max_idx + 1]]
    if not eligible:
        return None

    mastered = {w for w, c in correct_counts.items() if c >= MASTERED_THRESHOLD}

    # 計算每個單字距上次出現的題數（1 = 上一題）
    last_pos: dict = {}
    for i, w in enumerate(history):
        if w not in last_pos:
            last_pos[w] = i + 1

    def _ok(word):
        pos = last_pos.get(word)
        if pos is None:
            return True
        gap = MASTERED_GAP if word in mastered else NORMAL_GAP
        return pos >= gap

    new_ones = [w for w in eligible if w['word'] not in correct_counts and _ok(w['word'])]
    old_ones = [w for w in eligible if w['word'] in correct_counts
                and w['word'] not in mastered and _ok(w['word'])]
    all_pool = new_ones + old_ones

    # 全部冷卻中：放寬限制，取間隔最久的
    if not all_pool:
        candidates = [w for w in eligible if w['word'] not in mastered] or eligible
        candidates.sort(key=lambda w: last_pos.get(w['word'], 999), reverse=True)
        all_pool = candidates[:max(1, len(candidates) // 3)]

    # 最多嘗試 5 次，避免挖空失敗的題目出現
    for _attempt in range(5):
        if old_ones and random.random() > 0.7:
            target_raw = random.choice(old_ones)
        else:
            target_raw = random.choice(all_pool)

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

        is_verb = 'v.' in str(target.get('translation', ''))
        final_sentence, matched_tag = _find_and_blank(display_s, target['word'], is_verb)

        # 如果挖空成功（句子含底線）就使用，否則換一題
        if '______' in final_sentence:
            break
    else:
        # 5 次都失敗，直接用最後一個（至少不會卡死）
        pass

    if is_verb and matched_tag:
        answer_display = _inflect_word(target['word'], matched_tag)
    else:
        answer_display = re.sub(r'\(.*?\)', '', target['word'].split('/')[0]).strip()

    distractor_pool = [w for w in (VERB_POOL if is_verb else WORD_POOL)
                       if w['word'] != target['word']]
    distractors = random.sample(distractor_pool, min(3, len(distractor_pool)))

    options = []
    for d in distractors:
        dw = _inflect_word(d['word'], matched_tag) if (is_verb and matched_tag) else \
             re.sub(r'\(.*?\)', '', d['word'].split('/')[0]).strip()
        options.append({"word": dw, "base_word": d['word'], "trans": d['translation']})

    options.append({"word": answer_display, "base_word": target['word'], "trans": target['translation']})
    random.shuffle(options)

    return {
        "word":           target['word'],
        "answer_display": answer_display,
        "translation":    target['translation'],
        "level":          str(target.get('toeic_target', '')),
        "sentence":       final_sentence,
        "trans_s":        display_t,
        "options":        options,
        "verb_tag":       matched_tag,
    }


def index(request):
    # 傳完整單字資料給前端（單字表用）
    word_data = [
        {
            'word':        w['word'],
            'translation': w.get('translation', ''),
            'sentence':    w.get('sentence', ''),
            'trans_s':     w.get('trans_s', ''),
            'level':       str(w.get('toeic_target', '')),
        }
        for w in WORD_POOL
    ]
    return render(request, 'toeic_app/index.html', {
        'stats':       json.dumps(LEVEL_COUNTS),
        'level_words': json.dumps(LEVEL_WORDS),
        'word_data':   json.dumps(word_data),
    })


@require_GET
def get_question(request):
    difficulty  = request.GET.get('difficulty', '中級')
    history_raw = request.GET.get('history', '[]')
    counts_raw  = request.GET.get('counts', '{}')
    try:
        history = json.loads(history_raw)
    except Exception:
        history = []
    try:
        correct_counts = json.loads(counts_raw)
    except Exception:
        correct_counts = {}

    q = _get_question(difficulty, history, correct_counts)
    if not q:
        return JsonResponse({"error": "no questions"}, status=404)
    return JsonResponse(q)
