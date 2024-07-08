from fuzzywuzzy import fuzz, process

def process_duldul_list(duldul_list):
    # 사용자 딕셔너리 정의
    user_duldul = {
        '이민진': 'A',
        '정다운': 'B',
        '이재진': 'C',
        '제갈준영': 'D'
    }

    def find_best_match(extracted_text, user_duldul):
        best_match, score = process.extractOne(extracted_text, user_duldul.keys(), scorer=fuzz.token_sort_ratio)
        return best_match

    matched_list = []
    for text in duldul_list:
        best_match = find_best_match(text, user_duldul)
        matched_value = user_duldul[best_match]
        if not matched_list or matched_list[-1] != matched_value:
            matched_list.append(matched_value)
            if len(matched_list) == 3:
                break

    # 알파벳 순으로 정렬
    duldul_shiplist = sorted(matched_list)

    return duldul_shiplist
