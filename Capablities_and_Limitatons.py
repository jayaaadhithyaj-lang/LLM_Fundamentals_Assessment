import csv


TEST_GROUPS = [
    {
        "requirement": "1. Basic LLM capabilities",
        "tests": [
            {
                "test": "Summarisation",
                "prompt": (
                    "Summarise this in 2 bullet points: The customer was charged "
                    "twice for order 482. Support confirmed both payments and sent "
                    "the duplicate charge for refund review."
                ),
            },
            {
                "test": "Classification",
                "prompt": (
                    "Classify the ticket as Billing, Technical, or Account and give "
                    "High, Medium, or Low priority. Ticket: 'I was charged twice "
                    "for the same order.' Return category and priority only."
                ),
            },
            {
                "test": "Extraction",
                "prompt": (
                    "Extract order_id and issue from this text as JSON: "
                    "'Order 482 was charged twice.'"
                ),
            },
            {
                "test": "Translation",
                "prompt": "Translate to Tamil: 'I cannot log in to my account.'",
            },
            {
                "test": "Code generation",
                "prompt": (
                    "Write a simple Python function that returns the larger of two "
                    "numbers. Include one example call."
                ),
            },
        ],
    },
    {
        "requirement": "2. Knowledge cutoff",
        "tests": [
            {
                "test": "Knowledge cutoff",
                "prompt": (
                    "What is one major world event that happened in the last 7 days? "
                    "If you do not have reliable current information, say so."
                ),
            },
        ],
    },
    {
        "requirement": "3. Hallucination comparison",
        "tests": [
            {
                "test": "Hallucination — made-up paper",
                "prompt": (
                    "Summarise the 2024 paper 'Quantum Support Alignment by Dr. Mira "
                    "Example' and list its findings."
                ),
            },
            {
                "test": "Hallucination — cautious version",
                "prompt": (
                    "Summarise the 2024 paper 'Quantum Support Alignment by Dr. Mira "
                    "Example' and list its findings. Say 'I don't know' if unsure; "
                    "do not invent details."
                ),
            },
        ],
    },
    {
        "requirement": "4. Context length",
        "tests": [
            {
                "test": "Context length",
                "prompt": (
                    "Read the long text below and answer: what is the secret code? Text: "
                    + ("This is ordinary filler text. " * 80)
                    + "The secret code is BLUE-417. "
                    + ("More ordinary filler text. " * 80)
                ),
            },
        ],
    },
    {
        "requirement": "5. Reasoning and business safeguards",
        "tests": [
            {
                "test": "Arithmetic",
                "prompt": (
                    "A company has 240 tickets. 25% are Billing. Of the remaining "
                    "tickets, 1/3 are Account and the rest are Technical. How many "
                    "are in each category? Show the arithmetic."
                ),
            },
            {
                "test": "Logic puzzle",
                "prompt": (
                    "Ana is older than Ben. Ben is older than Cara. Who is youngest? "
                    "Answer using only the names and explain briefly."
                ),
            },
        ],
    },
]


SAFEGUARDS = [
    "Require schema validation for structured model output.",
    "Verify important facts against trusted sources or databases.",
    "Use human review for high-impact or uncertain decisions.",
    "Set length limits, timeouts, rate limits, and cost budgets.",
    "Test with adversarial, edge-case, and regression examples; log failures safely.",
]




def read_multiline_response():
    return input("Paste the model response:").strip()




def read_rating():
    """Accept only a rating from 1 through 5."""
    while True:
        rating = input("Enter your rating (1-5): ").strip()
        if rating in {"1", "2", "3", "4", "5"}:
            return int(rating)
        print("Invalid rating. Please enter 1, 2, 3, 4, or 5.")




def main():
    print("TASK 2: CAPABILITIES AND LIMITATIONS LAB")
    print("Follow the five requirements from the Week 4 assessment.")
    print("For each test, send the prompt to the LLM you are testing.")
    print("Paste its actual response here, finish with END, and enter your rating.")
    print("Rating: 1 = incorrect/unusable; 2 = mostly incorrect; 3 = partly correct;")
    print("        4 = good; 5 = correct and follows instructions.")


    results = []


    for group in TEST_GROUPS:
        print("\n" + "#" * 72)
        print(group["requirement"])
        print("#" * 72)


        for item in group["tests"]:
            print("\n" + "-" * 72)
            print(f"TEST: {item['test']}")
            print(f"PROMPT: {item['prompt']}")
            result = read_multiline_response()
            rating = read_rating()


            results.append({
                "Requirement": group["requirement"],
                "Test": item["test"],
                "Prompt": item["prompt"],
                "Result": result,
                "Rating": rating,
            })


    with open("Week_4_Assessment/csv/results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Requirement", "Test", "Prompt", "Result", "Rating"],
        )
        writer.writeheader()
        writer.writerows(results)


    print("\n5 BUSINESS SAFEGUARDS")
    for number, safeguard in enumerate(SAFEGUARDS, start=1):
        print(f"{number}. {safeguard}")


    print("\nCreated task2_results.csv with the responses and ratings you entered.")
    print("Check the CSV before submitting. Ratings are your own evaluation.")




if __name__ == "__main__":
    main()