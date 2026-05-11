import os
import re
import time
import random
import dashscope
from typing import Optional

from pydub import AudioSegment


MAX_API_RETRY = 10
API_RETRY_SLEEP = (1, 2)


language_code_mapping = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese"
}


language_alias_mapping = {
    "ar": "ar",
    "arabic": "ar",
    "cs": "cs",
    "czech": "cs",
    "da": "da",
    "danish": "da",
    "de": "de",
    "german": "de",
    "en": "en",
    "eng": "en",
    "english": "en",
    "es": "es",
    "spanish": "es",
    "fi": "fi",
    "finnish": "fi",
    "fil": "fil",
    "filipino": "fil",
    "fr": "fr",
    "french": "fr",
    "hi": "hi",
    "hindi": "hi",
    "id": "id",
    "indonesian": "id",
    "is": "is",
    "icelandic": "is",
    "it": "it",
    "italian": "it",
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "korean": "ko",
    "ms": "ms",
    "malay": "ms",
    "no": "no",
    "norwegian": "no",
    "pl": "pl",
    "polish": "pl",
    "pt": "pt",
    "portuguese": "pt",
    "ru": "ru",
    "russian": "ru",
    "sv": "sv",
    "swedish": "sv",
    "th": "th",
    "thai": "th",
    "tr": "tr",
    "turkish": "tr",
    "uk": "uk",
    "ukrainian": "uk",
    "vi": "vi",
    "vietnamese": "vi",
    "yue": "yue",
    "cantonese": "yue",
    "zh": "zh",
    "cn": "zh",
    "zh-cn": "zh",
    "mandarin": "zh",
    "putonghua": "zh",
    "chinese": "zh",
    "中文": "zh",
    "汉语": "zh",
    "普通话": "zh",
    "粤语": "yue",
    "英文": "en",
    "英语": "en",
    "日语": "ja",
    "德语": "de",
    "韩语": "ko",
    "俄语": "ru",
    "法语": "fr",
    "葡萄牙语": "pt",
    "阿拉伯语": "ar",
    "意大利语": "it",
    "西班牙语": "es",
    "印地语": "hi",
    "印尼语": "id",
    "泰语": "th",
    "土耳其语": "tr",
    "乌克兰语": "uk",
    "越南语": "vi",
    "捷克语": "cs",
    "丹麦语": "da",
    "菲律宾语": "fil",
    "芬兰语": "fi",
    "冰岛语": "is",
    "马来语": "ms",
    "挪威语": "no",
    "波兰语": "pl",
    "瑞典语": "sv"
}


def normalize_language_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None

    normalized = language.strip().lower().replace("_", "-")
    if not normalized:
        return None

    if normalized in language_alias_mapping:
        return language_alias_mapping[normalized]

    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,3})?", normalized):
        return normalized

    supported_codes = ", ".join(sorted(language_code_mapping))
    raise ValueError(
        f"Unsupported language '{language}'. Use one of the API language codes: {supported_codes}."
    )


class QwenASR:
    def __init__(self, model: str = "qwen3-asr-flash"):
        self.model = model

    def post_text_process(self, text, threshold=20):
        def fix_char_repeats(s, thresh):
            res = []
            i = 0
            n = len(s)
            while i < n:
                count = 1
                while i + count < n and s[i + count] == s[i]:
                    count += 1

                if count > thresh:
                    res.append(s[i])
                    i += count
                else:
                    res.append(s[i:i + count])
                    i += count
            return ''.join(res)

        def fix_pattern_repeats(s, thresh, max_len=20):
            n = len(s)
            min_repeat_chars = thresh * 2
            if n < min_repeat_chars:
                return s

            i = 0
            result = []
            while i <= n - min_repeat_chars:
                found = False
                for k in range(1, max_len + 1):
                    if i + k * thresh > n:
                        break

                    pattern = s[i:i + k]

                    valid = True
                    for rep in range(1, thresh):
                        start_idx = i + rep * k
                        if s[start_idx:start_idx + k] != pattern:
                            valid = False
                            break

                    if valid:
                        total_rep = thresh
                        end_index = i + thresh * k
                        while end_index + k <= n and s[end_index:end_index + k] == pattern:
                            total_rep += 1
                            end_index += k

                        result.append(pattern)
                        result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                        i = n
                        found = True
                        break

                if found:
                    break
                else:
                    result.append(s[i])
                    i += 1

            if not found:
                result.append(s[i:])
            return ''.join(result)

        text = fix_char_repeats(text, threshold)
        return fix_pattern_repeats(text, threshold)

    def asr(self, wav_url: str, context: str = "", language: Optional[str] = None):
        normalized_language = normalize_language_code(language)

        if not wav_url.startswith("http"):
            assert os.path.exists(wav_url), f"{wav_url} not exists!"
            file_path = wav_url
            file_size = os.path.getsize(file_path)

            # file size > 10M
            if file_size > 10 * 1024 * 1024:
                # convert to mp3
                mp3_path = os.path.splitext(file_path)[0] + ".mp3"
                audio = AudioSegment.from_file(file_path)
                audio.export(mp3_path, format="mp3")
                wav_url = mp3_path

            wav_url = f"file://{wav_url}"

        # Submit the ASR task
        for _ in range(MAX_API_RETRY):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"text": context},
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"audio": wav_url},
                        ]
                    }
                ]
                asr_options = {
                    "enable_lid": True,
                    "enable_itn": False
                }
                if normalized_language is not None:
                    asr_options["language"] = normalized_language

                response = dashscope.MultiModalConversation.call(
                    model=self.model,
                    messages=messages,
                    result_format="message",
                    asr_options=asr_options
                )

                if response.status_code != 200:
                    raise Exception(f"http status_code: {response.status_code} {response}")
                output = response['output']['choices'][0]

                recog_text = None
                if len(output["message"]["content"]):
                    recog_text = output["message"]["content"][0]["text"]
                if recog_text is None:
                    recog_text = ""

                lang_code = None
                if "annotations" in output["message"]:
                    lang_code = output["message"]["annotations"][0]["language"]
                if lang_code is None:
                    lang_code = normalized_language
                language = language_code_mapping.get(lang_code, "Not Supported")

                return language, self.post_text_process(recog_text)
            except Exception as e1:
                print(f"Error: {e1}")
                try:
                    print(f"Retry {_ + 1}...  {wav_url}\n{response}")
                    if response.code == "DataInspectionFailed":
                        print(f"DataInspectionFailed! Invalid input audio \"{wav_url}\"")
                        break
                except Exception as e2:
                    print(f"Retry {_ + 1}...  {wav_url}\n{e2}")
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"{wav_url} task failed!\n{response}")


if __name__ == "__main__":
    qwen_asr = QwenASR(model="qwen3-asr-flash")
    asr_text = qwen_asr.asr(wav_url="/path/to/your/wav_file.wav")
    print(asr_text)
