import re

def repl(m):
        value = m.group(1).strip()
        slot  = m.group(2).strip()
        return f"]\n[{value} -> {slot}]"
    
def normalize_multi_pairs(text: str) -> str:
    pattern = re.compile(
        r',\s*([^,\[\]\n]+?)\s*->\s*([^,\[\]\n]+?)\s*]'
    )

    while pattern.search(text):
        text = pattern.sub(repl, text)
    return text

if __name__ == "__main__":
    ans = normalize_multi_pairs("[zapata -> restaurant_name, four -> party_size_number]")
    print(ans)