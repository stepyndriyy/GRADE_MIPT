
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, BlenderbotForConditionalGeneration


class DialogGenerator:
    def __init__(self):
        self.mname = None
        self.context = []
        return
    
    def clear_history(self):
        self.context = []
    
    def update_history(self, history: list[str]):
        self.context += history
    
    def generate_response(self, next_message: str = None):
        pass
    
    def print_dialog(self):
        print(*self.context)


class DialogGeneratorBlenderBot400(DialogGenerator):
    def __init__(self):
        super().__init__()
        self.mname = "facebook/blenderbot-400M-distill"
        #self.config = BlenderbotSmallConfig(max_new_tokens=1000)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.mname)
        self.tokenizer = AutoTokenizer.from_pretrained(self.mname)
        self.model.max_new_tokens = 1000 # TODO
        self.model.max_length = 1000
        self.message_separator = "</s> <s>"
        return

    def generate_response(self, next_message: str = None):
        if next_message is not None:
            self.context.append(next_message)

        previous_dialog = self.message_separator.join(self.context)

        inputs = self.tokenizer([previous_dialog], return_tensors="pt")

        reply_ids = self.model.generate(max_length=1000, **inputs)
        self.context.append(self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])

        return self.context[-1]
    
    def print_dialog(self):
        return self.message_separator.join(self.context)
    


class DialogGeneratorDialoGPT_medium(DialogGenerator):
    def __init__(self):
        super().__init__()
        
        self.mname = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(self.mname, padding_side='right')
        self.model = AutoModelForCausalLM.from_pretrained(self.mname)
    
        self.chat_history_ids = None

        return
    
    def clear_history(self):
        context = []
        self.chat_history_ids = None
        return
    
    def update_history(self, history: list[str]):
        super().update_history(history)
        for message in history:
            new_user_input_ids = self.tokenizer.encode(message + self.tokenizer.eos_token, return_tensors='pt')
            self.chat_history_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.chat_history_ids is not None else new_user_input_ids
        return 
    
    def generate_response(self, next_message=None):
        if (next_message == None):
            return
            
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(next_message + self.tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.chat_history_ids is not None else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        return self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


class BotsDialogGenerator(DialogGenerator):
    def __init__(self, speaker_a: DialogGenerator, speaker_b: DialogGenerator):
        super().__init__()

        self.speaker_a = speaker_a
        self.speaker_b = speaker_b
        return
    
    # TODO this is cinda shhhhiiiiiitttttt
    # deprecated
    def create_dialog(self, max_len=5, starting_message="Hi! my name Alice, tell me about yourself") -> list[str]:
        self.speaker_a.clear_history()
        self.speaker_b.clear_history()
        
        self.speaker_b.update_history([starting_message])
        next_message = starting_message
        self.dialog.append(next_message)
        
        for dialog_len in range(max_len):
            response = self.speaker_a.generate_response(next_message)
            self.dialog.append(response)

            next_message = self.speaker_b.generate_response(response)
            self.dialog.append(next_message)
            
        return self.dialog
    
    def init_dialog(self, starting_message="Hello neighbour!"):
        self.speaker_a.clear_history()
        self.speaker_b.clear_history()
        self.speaker_b.update_history([starting_message])
        self.context.append(starting_message)
    
    def generate_response(self) -> list[str]:
        self.context.append(self.speaker_a.generate_response(self.context[-1]))
        self.context.append(self.speaker_b.generate_response(self.context[-1]))
        return [self.context[-2], self.context[-1]]

