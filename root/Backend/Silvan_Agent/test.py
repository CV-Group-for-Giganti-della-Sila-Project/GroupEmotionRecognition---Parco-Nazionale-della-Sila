#!/usr/bin/env python3
import json
from silvan_agent import handle_request

def main():
    print("🤖 MySQL Insight Agent Initialized.")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 40)

    while True:
        try:
            # 1. Get the question from the user
            user_input = input("\nAsk a question: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue

            # 2. Build the payload
            payload = {
                "question": user_input,
                "debug": True  # Keep debug True to see the SQL it generates
            }

            # 3. Call your agent
            print("\nThinking...")
            response = handle_request(payload)

            # 4. Display the results
            print("\n" + "=" * 40)
            print("🧠 ANSWER:")
            print(response.get("answer", "No answer generated."))
            print("-" * 40)
            
            if "sql" in response:
                print("💻 SQL EXECUTED:")
                print(response["sql"])
                print("PARAMETERS:", response.get("params"))
            print("=" * 40)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()