# Start an infinite loop to keep the chat going until the user stops it.
    while True:
        # Use a 'try...except' block to gracefully handle unexpected errors during the chat.
        try:
            # Ask the user for their question and remove any extra spaces.
            question = input("\nYou: ").strip()
            # If the user just pressed Enter (empty question), skip the rest of the loop.
            if not question: 
                continue
                
            # Check if the user wants to quit the chat.
            if question.lower() in ["exit", "quit"]:
                # Exit the 'while True' loop, which ends the program.
                break

            print("Searching notes...")
            # Use the search function to find the most relevant notes (context) for the question.
            context = search_context(question, docs)
            
            print("Thinking...")
            # Use the ask function to get the final answer from the LLM, using the question,
            # the relevant notes (context), and the chat history.
            answer = ask_gemini(question, context, chat_history)

            # Print the AI's final answer.
            print(f"\nAI: {answer}")
            print("-" * 30)

            # Add the current user question and the AI's answer to the chat history list.
            chat_history.append({"user": question, "ai": answer})
            
            # This is a simple way to limit the size of the history to the last 5 turns.
            # If the list gets too long (more than 5), remove the oldest turn (the first item).
            if len(chat_history) > 5:
                chat_history.pop(0)
            
        # If any error occurs inside the loop, print an error message but keep the chat running.
        except Exception as e:
            print(f"\nAn error occurred: {e}")
