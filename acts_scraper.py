# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
from hashlib import sha1
import pandas as pd
import time
import re

# ✨ 추가: canonical text 변환용
from bs4 import BeautifulSoup
from textwrap import wrap

# ===================== 설정 =====================
CHUNK_OUTPUT = True          # 청크 CSV까지 만들고 싶다면 True
CHUNK_MAX_CHARS = 1800        # 권장 1200~2000
WINDOW_SIZE = "1280,1600"

# ===================== 드라이버 설정 =====================
options = webdriver.ChromeOptions()
# options.add_argument("--headless=new")  # 화면 없이 실행하고 싶다면 주석 해제
options.add_argument(f"--window-size={WINDOW_SIZE}")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 12)

# ===================== 유틸 함수들 =====================
def norm(s: str | None) -> str | None:
    if not s:
        return None
    return re.sub(r"\s+", " ", s).strip() or None

def get_text_css(drv, sel):
    try:
        return norm(drv.find_element(By.CSS_SELECTOR, sel).text)
    except NoSuchElementException:
        return None

def find_content_container(drv):
    """
    Justice Laws 컨텐츠 컨테이너 탐색.
    우선순위: div.wb-txthl > main div.wb-txthl > article div.wb-txthl > main > (fallback) body
    """
    selectors = [
        "div.wb-txthl",
        "main div.wb-txthl",
        "article div.wb-txthl",
        "main[role='main'] div.wb-txthl",
        "main",
    ]
    for sel in selectors:
        try:
            el = drv.find_element(By.CSS_SELECTOR, sel)
            if el and el.text.strip():
                return el
        except NoSuchElementException:
            continue
    # 최후의 보루
    return drv.find_element(By.TAG_NAME, "body")

def safe_get_page1_and_wait(list_url):
    """
    법령 목록(acts 상세) 페이지에서 Page 1로 진입하고 본문 로딩까지 대기.
    상대경로 href 보정(urljoin) 포함.
    """
    page1_link_el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[title='Page 1']")))
    href = page1_link_el.get_attribute("href")
    page1_url = urljoin(list_url, href)
    driver.get(page1_url)
    # 본문 컨테이너 또는 제목 등장 대기
    wait.until(
        EC.any_of(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.wb-txthl")),
            EC.presence_of_element_located((By.CSS_SELECTOR, "h2.Title-of-Act")),
            EC.presence_of_element_located((By.CSS_SELECTOR, "p.LongTitle"))
        )
    )
    return page1_url

# ======= HTML -> Canonical Text (RAG 표준 텍스트) =======
def container_to_canonical_text(driver) -> str:
    """
    컨테이너 innerHTML을 메모리에서만 파싱하여 RAG용 '공통 텍스트'로 정규화.
    - 제거: 개정연혁/모달/이전버전/숨김/script/style
    - 링크: 앵커 언랩(텍스트만)
    - 헤딩: #, ##, ### ... 라벨
    - 정의목록: '용어 — 정의' 한 줄화
    - 조문번호(.lawlabel) 래퍼 제거(번호 텍스트는 보존)
    """
    el = find_content_container(driver)
    html = el.get_attribute("innerHTML") or ""
    soup = BeautifulSoup(html, "html.parser")

    # 1) 제거 블록
    for sel in [".HistoricalNote", ".PITLink", ".modal-dialog", ".mfp-hide", "script", "style", ".wb-invisible"]:
        for node in soup.select(sel):
            node.decompose()

    # 2) 링크 언랩
    for a in soup.find_all("a"):
        a.unwrap()

    # 3) 헤딩 라벨링(# ~ #####)
    for level in range(1, 6):
        for h in soup.find_all(f"h{level}"):
            txt = h.get_text(" ", strip=True)
            h.string = f"{'#'*level} {txt}"

    # 4) 조문 번호 래퍼 제거
    for span in soup.select(".lawlabel"):
        span.unwrap()

    # 5) 줄바꿈 보정
    for br in soup.select("br"):
        br.replace_with("\n")

    # 6) 정의 목록 dt/dd -> 'term — definition'
    for dt in soup.find_all("dt"):
        term = dt.get_text(" ", strip=True)
        dd = dt.find_next_sibling("dd")
        if dd:
            dd_txt = dd.get_text(" ", strip=True)
            dt.string = f"{term} — {dd_txt}"
            dd.decompose()

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3줄 이상 → 2줄
    text = re.sub(r"[ \t]+\n", "\n", text)      # 줄끝 공백 제거
    return text

def text_sha1(text: str) -> str:
    return sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()

def chunk_text(txt: str, max_chars=1800):
    """단락(빈 줄) 기반으로 우선 분할 후, 초과 블록 하드랩."""
    blocks = re.split(r"\n{2,}", txt)
    chunks, buf = [], ""
    for b in blocks:
        if not b.strip():
            continue
        if len(buf) + len(b) + 2 <= max_chars:
            buf = (buf + "\n\n" + b) if buf else b
        else:
            if buf:
                chunks.append(buf)
            if len(b) <= max_chars:
                buf = b
            else:
                for w in wrap(b, width=max_chars, break_long_words=False, replace_whitespace=False):
                    if w.strip():
                        chunks.append(w)
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks

# ===================== 메인 로직 =====================
rows_full = []
rows_chunks = []

try:
    # 1) Acts 인덱스 접속
    index_url = "https://laws-lois.justice.gc.ca/eng/acts/"
    driver.get(index_url)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "statRes")))

    # 2) ul.statRes 안의 li 들에서 '첫 번째 a'의 href 수집
    ul_element = driver.find_element(By.CLASS_NAME, "statRes")
    li_elements = ul_element.find_elements(By.TAG_NAME, "li")

    hrefs = []
    for li in li_elements:
        try:
            a_tag = li.find_element(By.TAG_NAME, "a")
            href = a_tag.get_attribute("href")
            if href:
                hrefs.append(href)
        except NoSuchElementException:
            continue

    print(f"총 {len(hrefs)}개의 법령 링크 수집")

    # 3) 각 법령 상세 → Page 1 → canonical text 수집
    for idx, link in enumerate(hrefs, start=1):
        try:
            driver.get(link)
            print(f"[{idx}/{len(hrefs)}] 진입: {link}")

            try:
                page1_url = safe_get_page1_and_wait(link)
            except (TimeoutException, NoSuchElementException):
                print("   → Page 1 이동/로딩 실패, 현재 페이지에서 수집")
                page1_url = driver.current_url

            # 메타(있으면) + canonical text 생성
            title = get_text_css(driver, "h2.Title-of-Act")
            identifier = get_text_css(driver, "p.ChapterNumber")
            long_title = get_text_css(driver, "p.LongTitle")
            assented_date = get_text_css(driver, "p.AssentedDate")

            canonical_text = container_to_canonical_text(driver)
            canonical_sha1 = text_sha1(canonical_text)

            rows_full.append({
                "act_list_url": link,
                "page1_url": page1_url,
                "title": title,
                "identifier": identifier,
                "assented_date": assented_date,
                "long_title": long_title,
                "canonical_sha1": canonical_sha1,
                "canonical_text_full": canonical_text
            })

            if CHUNK_OUTPUT and canonical_text:
                chunks = chunk_text(canonical_text, max_chars=CHUNK_MAX_CHARS)
                for j, ch in enumerate(chunks):
                    rows_chunks.append({
                        "title": title,
                        "identifier": identifier,
                        "assented_date": assented_date,
                        "source_url": page1_url,
                        "chunk_index": j,
                        "text": ch
                    })

            # 너무 빠른 요청 방지
            time.sleep(0.3)

        except Exception as e:
            print(f"   ! 예외 발생, 스킵: {e}")

    # 4) 저장 (HTML 저장 없음, 텍스트만 저장)
    df_full = pd.DataFrame(rows_full)
    df_full.to_csv("canada_acts_page1_canonical.csv", index=False, encoding="utf-8-sig")
    print("저장 완료: canada_acts_page1_canonical.csv")

    if CHUNK_OUTPUT:
        df_chunks = pd.DataFrame(rows_chunks)
        df_chunks.to_csv("canada_acts_page1_canonical_chunks.csv", index=False, encoding="utf-8-sig")
        print("저장 완료: canada_acts_page1_canonical_chunks.csv")

finally:
    driver.quit()
