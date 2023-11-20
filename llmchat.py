from colorama import init, Style, Fore, Back

class LLMChat:
    def __init__(self, get_model_response_function) -> None:
        self.chat_history = None
        self.get_model_response = get_model_response_function

    def get_user_input(self) -> str:
        """Simple CLI for chat. Supports single-line and multi-line input as well as some additional features
        [HELP] - print help message

        [RAW] - show raw chat history

        [BEGIN] - start multiline input

        [END] - end multiline input

        [NEW] - forget previous conversation and start a new one
        
        [BEGIN_RAW] - raw chat history input start
        [END_RAW] - raw chat history input finish
        
        [GO] - execute current chat history

        Returns:
            str: user input
        """
        is_multiline = False

        while True:
            if not is_multiline:
                print(Fore.LIGHTCYAN_EX + "USR:>", end="")

            input_line = input()

            match(input_line):
                case "":
                    continue
                case "[HELP]":
                    print(Fore.LIGHTBLUE_EX + "Type your text to converse.\nMultiline mode ON: [BEGIN]\nMultiline mode OFF: [END]\nRaw chat history: [RAW]")
                    continue
                case "[RAW]":
                    if self.chat_history is not None:
                        print(Fore.LIGHTGREEN_EX + self.chat_history)
                    continue
                case "[NEW]":
                    print(Fore.LIGHTRED_EX + "!!!New conversation started. All previous history is forgotten!!!")
                    self.chat_history = None
                    continue
                case "[BEGIN]":
                    user_text = ""
                    is_multiline = True
                    continue
                case "[END]":
                    return user_text
                case "[BEGIN_RAW]":
                    print(Fore.LIGHTYELLOW_EX + "RAW CHAT HISTORY INPUT: ON")
                    user_text = ""
                    is_multiline = True
                    continue
                case "[END_RAW]":
                    print(Fore.LIGHTYELLOW_EX + "RAW CHAT HISTORY INPUT: OFF")
                    self.chat_history = user_text
                    is_multiline = False
                    continue
                case "[GO]":
                    return ""
                case _:
                    if is_multiline:
                        user_text += input_line + "\n"
                    else:
                        return input_line

    def get_system_message(self) -> str:
        """Simple CLI to get model system message

        [HELP] - print help message

        [BEGIN] - start multiline input

        [END] - end multiline input

        [DEFAULT] - use model's default system message

        Returns:
            str: system message string or None
        """

        is_multiline = False
        while True:

            if not is_multiline:
                print(Fore.LIGHTYELLOW_EX + "Prompt System message:", end="")

            input_txt = input()

            match(input_txt):
                case "[HELP]":
                    print(Fore.LIGHTBLUE_EX + "Single-line input mode\nMultiline mode ON: [BEGIN]\nMultiline mode OFF: [END]\nDefault model system message: [DEFAULT]")
                    continue
                case "[BEGIN]":
                    system_msg = ""
                    is_multiline = True
                    continue
                case "[END]":
                    return system_msg
                case "[DEFAULT]":
                    return None
                case _:
                    if is_multiline:
                        system_msg += input_txt + "\n"
                    else:
                        return input_txt
