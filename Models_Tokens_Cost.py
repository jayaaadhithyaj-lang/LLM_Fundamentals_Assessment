import json
import tiktoken


MODEL_FAMILIES = [
    ["GPT", "OpenAI", "Closed", "General tasks, coding"],
    ["Claude", "Anthropic", "Closed", "Writing, analysis, coding"],
    ["Llama", "Meta", "Open-weight", "Self-hosting, customization"],
    ["Gemini", "Google", "Closed", "Multimodal tasks"],
]


SAMPLE_TEXTS = {
    "Sentence": "The support team solved my login issue today.",
    "Paragraph": (
        "A customer could not sign in after resetting a password. "
        "Support checked the account, reset the session, and asked the customer to try again."
    ),
    "Page": ("This is a sample page for token counting. " * 100).strip(),
    "JSON": json.dumps({
        "ticket_id": 1042,
        "category": "Technical",
        "priority": "High",
        "message": "The application crashes when I upload a file."
    }),
    "Tamil sentence": "வணக்கம், எனக்கு கணக்கில் உள்நுழைய உதவி தேவை."
}




# Function to print a table
def print_table(headers, rows):
    widths = []


    for i in range(len(headers)):
        width = len(str(headers[i]))


        for row in rows:
            if len(str(row[i])) > width:
                width = len(str(row[i]))


        widths.append(width)


    line = "+"
    for width in widths:
        line = line + "-" * (width + 2) + "+"
    print(line)


    print("|", end="")
    for i in range(len(headers)):
        print(" " + str(headers[i]).ljust(widths[i]) + " |", end="")
    print()


    print(line)


    for row in rows:
        print("|", end="")
        for i in range(len(row)):
            print(" " + str(row[i]).ljust(widths[i]) + " |", end="")
        print()


    print(line)




# Count tokens
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))




# Calculate cost
def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    cost = (input_tokens / 1000000 * price_in)
    cost = cost + (output_tokens / 1000000 * price_out)
    return cost




def main():


    # TASK 1: MODEL FAMILY COMPARISON
    print("\nTASK 1: MODEL FAMILY COMPARISON")


    print_table(
        ["Family", "Owner", "Weights", "Strengths"],
        MODEL_FAMILIES
    )


    # TASK 2: TOKEN COUNTS
    print("\nTASK 2: TOKEN COUNTS")


    token_counts = {}
    token_rows = []


    for name, text in SAMPLE_TEXTS.items():
        words = len(text.split())
        tokens = count_tokens(text)


        token_counts[name] = tokens


        if tokens > 0:
            ratio = round(words / tokens, 2)
        else:
            ratio = 0


        token_rows.append([name, words, tokens, ratio])


    print_table(
        ["Text Type", "Words", "Tokens", "Words/Token"],
        token_rows
    )


    # TASK 3: CONTEXT WINDOW CHECK
    print("\nTASK 3: CONTEXT WINDOW CHECK")


    context_rows = []


    for name, tokens in token_counts.items():
        if tokens <= 8000:
            fit_8k = "Yes"
        else:
            fit_8k = "No"


        if tokens <= 128000:
            fit_128k = "Yes"
        else:
            fit_128k = "No"


        context_rows.append(
            [name, tokens, fit_8k, fit_128k]
        )


    print_table(
        ["Text Type", "Tokens", "Fits 8K", "Fits 128K"],
        context_rows
    )


    # TASK 4: MONTHLY COST ESTIMATE
    print("\nTASK 4: MONTHLY COST ESTIMATE")


    chats_per_day = 1000
    days_per_month = 30
    input_per_chat = 1500
    output_per_chat = 400


    monthly_input = chats_per_day * days_per_month * input_per_chat
    monthly_output = chats_per_day * days_per_month * output_per_chat


    # Example prices
    price_in = 1.00
    price_out = 3.00


    monthly_cost = estimate_cost(
        monthly_input,
        monthly_output,
        price_in,
        price_out
    )


    cost_rows = [
        ["Monthly Input Tokens", monthly_input],
        ["Monthly Output Tokens", monthly_output],
        ["Input Price per 1M Tokens", price_in],
        ["Output Price per 1M Tokens", price_out],
        ["Estimated Monthly Cost", round(monthly_cost, 2)]
    ]


    print_table(["Item", "Value"], cost_rows)


    # TASK 5: RECOMMENDATIONS
    print("\nTASK 5: RECOMMENDATIONS")


    recommendation_rows = [
        [
            "Public Support Bot",
            "Hosted GPT or Gemini",
            "Easy deployment and scalability"
        ],
        [
            "Private On-Premise Tool",
            "Llama",
            "More control over data and customization"
        ]
    ]


    print_table(
        ["Use Case", "Suggested Model", "Reason"],
        recommendation_rows
    )




if __name__ == "__main__":
    main()