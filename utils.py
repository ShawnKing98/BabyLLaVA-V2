import re
import base64
import matplotlib.pyplot as plt
import json
import ipdb
import torch
import torch.nn as nn
import numpy as np

def _reinit_module_parameters(module: nn.Module):
    # 尽可能通用地重置
    if hasattr(module, "reset_parameters"):
        try:
            module.reset_parameters()
            return
        except Exception:
            pass
    # 常见层的兜底
    if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.LayerNorm)):
        for p in module.parameters():
            if p.requires_grad:
                nn.init.normal_(p, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

def reinit_all_parameters(model: nn.Module, verbose=True):
    for name, module in model.named_modules():
        _reinit_module_parameters(module)
    # 某些模型会在 class 里定义 _init_weights
    if hasattr(model, "_init_weights"):
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() > 1:
                model._init_weights(p)
    if verbose:
        print("All parameters re-initialized.")

def encode_image_to_base64(path):
    import base64, mimetypes
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# for counting task
SPELL_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}
NUM_TO_SPELL = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"
}

# from arabic numbers to word numbers, according to SPELL_TO_NUM
def extract_word_number(text):
    """Return the first spelled-out number found in text, or None if none found."""
    if isinstance(text, str):
        return NUM_TO_SPELL.get(text.lower())
    
    


def extract_number(text):
    """Return the first number (digit or spelled-out) found in text, or None if none found."""
    if not text:
        return None
    # regex finds digits or words that match SPELL_TO_NUM
    if isinstance(text, str):
        tokens = re.findall(r"\d+|[a-zA-Z]+", text.lower())
        for tok in tokens:
            if tok.isdigit():
                return int(tok)
            if tok in SPELL_TO_NUM:
                return SPELL_TO_NUM[tok]
    elif isinstance(text, (int, float)):
        return int(text)
    return None

def extract_letter(text):
    """Return the first letter (A, B, C, D) found in text, or None if none found."""
    if not text:
        return None
    if isinstance(text, str):
        match = re.search(r'\b([A-Da-d])\b', text)
        if match:
            return match.group(1).upper()
    return None


# for visual delay response task
# valid outputs
valid_direction_words = ['left', 'right', 'top', 'bottom', 'top right', 'top left', 'bottom right', 'bottom left']

# adjacent directions for open evaluation
adjacent_directions_word_mapping = {
        'left': ['top left', 'left', 'bottom left'],
        'right': ['top right', 'right', 'bottom right'],
        'top': ['top left', 'top', 'top right'],
        'bottom': ['bottom left', 'bottom', 'bottom right'],
        'top right': ['top', 'top right', 'right'],
        'top left': ['top', 'top left', 'left'],
        'bottom right': ['bottom', 'bottom right', 'right'],
        'bottom left': ['bottom', 'bottom left', 'left']
    }

# map label directions to correct words
directions_new_word_mapping = {
        'Up': 'top',
        'Down': 'bottom',
        'Left': 'left',
        'Right': 'right'
}

# if 2 directions, then map to combined one
direction_combinations_mapping = {
    frozenset(['top', 'right']): 'top right',
    frozenset(['top', 'left']): 'top left',
    frozenset(['bottom', 'right']): 'bottom right',
    frozenset(['bottom', 'left']): 'bottom left',
}

# opposite direction
opposite_direction_word_mapping = {
        'left': 'right',
        'right': 'left',
        'top': 'bottom',
        'bottom': 'top',
        'top right': 'bottom left',
        'top left': 'bottom right',
        'bottom right': 'top left',
        'bottom left': 'top right'
}


# create system prompt
def create_qwen_system_prompt(prompt=None):
    """Create system prompt for the given category"""
    if prompt == "memory":
        return f"""<|im_start|>system
You are administering a  memory test.  \\
First you will be shown 1 image and then each round you will be shown 2 images, one of which is new which you have not seen before, the other is a repeat one from earlier rounds. \\
Keep track of every image you have seen so far. \\
When the user says "touch the new image" respond _only_ with the letter of option that contains the novel image without any other text, such as "A". """
    elif prompt == "count":
        return f"""<|im_start|>system
You are administering a counting test.  \\
Count the objects in the image. Reply _only_ with the number of objects in the image, such as '4'. """
    elif prompt == "colorsize":
        return f"""<|im_start|>system
You are administering a color and size recognition test. \\
In each round, you will be shown one image with a stimulus object in the center and another image with 4 objects of different colors or sizes. \\
Respond _only_ with the number(1-4) of the option that matches the color or size of the stimulus object without any other text, such as "2". """
    elif prompt == "leftright":
        return f"""<|im_start|>system
You are administering a left-right spatial recognition test. \\
In each round, you will be shown one image with a stimulus object and another 3 images with 3 objects in different left-right positions. \\
Respond _only_ with the letter(A-C) of the option that matches the left-right position of the stimulus object without any other text, such as "B". """
    elif prompt == "spatialdetails":
        return f"""<|im_start|>system
You are administering a spatial details recognition test. \\
In each round, you will be shown one image with a stimulus object and another 3 images with 3 different objects. \\
Respond _only_ with the letter(A-C) of the option that matches the object in the stimulus image without any other text, such as "B". """
    elif prompt == "picture_vocabulary":
        return f"""<|im_start|>system
You are administering a picture vocabulary test. \\
In each round, you will be given one label and shown 4 images with 4 different objects. \\
Respond _only_ with the letter(A-D) of the option that matches the label without any other text, such as "A". """
    elif prompt == "localize":
        return f"""<|im_start|>system
You are administering a spatial localization test. \\
In each round, you will be given one object label with an image. \\
Respond _only_ with the letter(A-D) of the option that matches the location of the object in the image without any other text, such as "C". """
    elif prompt == "point_count":
        return f"""<|im_start|>system
You are administering a point counting test. \\
In each round, you will be given one object label and shown one image with several objects in it. \\
Respond _only_ with the number of objects that match the label in the image, such as "4". """
    elif prompt == "compare_synthetic" or prompt == "compare_real":
        return f"""<|im_start|>system
You are administering a visual comparison test. \\
In each round, you will be shown two images and asked to compare them based on object number. \\
Respond _only_ with the letter(A/B) of the option image that has more objects without any other text, such as "A". """
    else:
        return f"""<|im_start|>system
You are a helpful assistant."""

# api version system prompt, only text
def create_api_system_prompt(prompt=None):
    """Create system prompt for the given category"""
    if prompt == "memory":
        return f"""You are administering a memory test.  \\
First you will be shown 1 image and then each round you will be shown 2 images, one of which is new which you have not seen before, the other is a repeat one from earlier rounds. \\
Keep track of every image you have seen so far. \\
When the user says "touch the new image" respond _only_ with the letter of option that contains the novel image without any other text, such as "A". """
    elif prompt == "count":
        return f"""You are administering a counting test.  \\
Count the objects in the image. Reply _only_ with the number of objects in the image, such as '4'. """
    elif prompt == "colorsize":
        return f"""You are administering a color and size recognition test. \\
In each round, you will be shown one image with a stimulus object in the center and another image with 4 objects of different colors or sizes. \\
Respond _only_ with the number(1-4) of the option that matches the color or size of the stimulus object without any other text, such as "2". """
    elif prompt == "leftright":
        return f"""You are administering a left-right spatial recognition test. \\
In each round, you will be shown one image with a stimulus object and another 3 images with 3 objects in different left-right positions. \\
Respond _only_ with the letter(A-C) of the option that matches the left-right position of the stimulus object without any other text, such as "B". """
    elif prompt == "spatialdetails":
        return f"""You are administering a spatial details recognition test. \\
In each round, you will be shown one image with a stimulus object and another 3 images with 3 different objects. \\
Respond _only_ with the letter(A-C) of the option that matches the object in the stimulus image without any other text, such as "A". """
    elif prompt == "picture_vocabulary":
        return f"""You are administering a picture vocabulary test. \\
In each round, you will be given one label and shown 4 images with 4 different objects. \\
Respond _only_ with the letter(A-D) of the option that matches the label without any other text, such as "A". """
    elif prompt == "localize":
        return f"""You are administering a spatial localization test. \\
In each round, you will be given one object label with an image. \\
Respond _only_ with the letter(A-D) of the option that matches the location of the object in the image without any other text, such as "C". """
    elif prompt == "point_count":
        return f"""You are administering a point counting test. \\
In each round, you will be given one object label and shown one image with several objects in it. \\
Respond _only_ with the number of objects that match the label in the image, such as "4". """
    elif prompt == "compare_synthetic" or prompt == "compare_real":
        return f"""You are administering a visual comparison test. \\
In each round, you will be shown two images and asked to compare them based on object number. \\
Respond _only_ with the letter(A/B) of the option image that has more objects without any other text, such as "A". """
    else:
        return f"""You are a helpful assistant."""


###########################
# TASK ACCURACY CALCULATION
###########################

def calc_memory_task_accuracy(predictions, sample):
    """Calculate accuracy for memory task"""
    base_round = (len(predictions)+1) // 3
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = chat_round["value"].strip().split()[-1] if chat_round["value"].strip() else ""
            ground_truths.append(extract_letter(answer))
            # ground_truths.append(answer)
    
    predictions = [extract_letter(p) for p in predictions]

    # Calculate accuracy for learning phase (first base_round rounds)
    learning_phase_preds = predictions[:base_round]
    learning_phase_gts = ground_truths[:base_round]
    learning_accuracy = sum([p == gt for p, gt in zip(learning_phase_preds, learning_phase_gts)]) / len(learning_phase_gts) if learning_phase_gts else 0.0

    # Calculate raw accuracy for testing phase (last base_round*2 rounds)
    testing_phase_preds = predictions[base_round:]
    testing_phase_gts = ground_truths[base_round:]
    testing_raw_accuracy = sum([p == gt for p, gt in zip(testing_phase_preds, testing_phase_gts)]) / len(testing_phase_gts) if testing_phase_gts else 0.0

    # calculate adjusted accuracy for testing phase 
    # each old image appears twice, so if the model gets two corrects, we count it as correct
    testing_adjusted_correct = 0
    
    # Use labels field to find which testing rounds reference the same learning phase labels
    labels = sample["labels"]
    learning_labels = set()
    
    # Collect all labels from learning phase (first base_round rounds)
    for i in range(base_round):
        learning_labels.update(labels[i])
    
    # Group testing phase rounds by shared learning labels
    label_to_rounds = {}  # {label: [round_indices]}
    
    # Process testing phase rounds (from base_round onwards)
    for i in range(base_round, len(labels)):
        test_round_idx = i - base_round  # 0-based index in testing phase
        round_labels = set(labels[i])
        
        # Find which learning label appear in this testing round
        # if there is no shared label, skip this
        shared_labels = list(round_labels.intersection(learning_labels))
        if shared_labels == []:
            continue
        # We assume there is only one shared label between learning and testing phase
        label_to_rounds.setdefault(shared_labels[0], []).append(test_round_idx)
    
    # if label_to_rounds is empty, return 0.0
    if label_to_rounds == {}:
        testing_adjusted_accuracy = 0.0
    else:
        # Count pairs where both rounds are correct
        for label, round_indices in label_to_rounds.items():
            if len(round_indices) == 2:  # Should have exactly 2 occurrences per old label
                round1_idx, round2_idx = round_indices
                if (round1_idx < len(testing_phase_preds) and round2_idx < len(testing_phase_preds) and
                    testing_phase_preds[round1_idx] == testing_phase_gts[round1_idx] and
                    testing_phase_preds[round2_idx] == testing_phase_gts[round2_idx]):
                    testing_adjusted_correct += 1
        
        testing_adjusted_accuracy = testing_adjusted_correct / base_round if base_round > 0 else 0.0
    
    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1
    
    return {
        "learning_accuracy": learning_accuracy,
        "testing_raw_accuracy": testing_raw_accuracy,
        "testing_adjusted_accuracy": testing_adjusted_accuracy,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }
  
def sum_memory_task_accuracies(accuracies_list):
    """Summarize accuracies over multiple samples"""
    total_learning_acc = 0.0
    total_testing_raw_acc = 0.0
    total_testing_adjusted_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_learning_acc += acc["learning_accuracy"]
        total_testing_raw_acc += acc["testing_raw_accuracy"]
        total_testing_adjusted_acc += acc["testing_adjusted_accuracy"]
    
    return {
        "avg_learning_accuracy": total_learning_acc / num_samples if num_samples > 0 else 0.0,
        "avg_testing_raw_accuracy": total_testing_raw_acc / num_samples if num_samples > 0 else 0.0,
        "avg_testing_adjusted_accuracy": total_testing_adjusted_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_count_task_accuracy(predictions, sample):
    base_round = len(predictions) // 2
    
    # use extract_number to get the first number from model response and ground truth
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = extract_number(chat_round["value"])
            ground_truths.append(answer)
            
    # match predictions with ground truths
    preds = [extract_number(p) for p in predictions]
    acc = sum([p == gt for p, gt in zip(preds, ground_truths)]) / len(ground_truths) if ground_truths else 0.0
    
    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1
    
    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }

def sum_count_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]
    
    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_vdr_task_accuracy(predictions, sample):
    base_round = len(predictions) // 2
    
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = chat_round["value"].strip() if chat_round["value"].strip() else ""
            ground_truths.append(answer)
            
    # match exactly
    accs_exact = [p.lower() == gt.lower() for p, gt in zip(predictions, ground_truths)]
    acc_exact = sum(accs_exact) / len(accs_exact) if accs_exact else 0.
    
    # match adjacent directions
    accs_adjacent = [p.lower() in adjacent_directions_word_mapping.get(gt.lower(), []) for p, gt in zip(predictions, ground_truths)]
    acc_adjacent = sum(accs_adjacent) / len(accs_adjacent) if accs_adjacent else 0.

    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1

    return {
        "accuracy_exact": acc_exact,
        "accuracy_adjacent": acc_adjacent,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }
    
def sum_vdr_task_accuracies(accuracies_list):
    total_acc_exact = 0.0
    total_acc_adjacent = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc_exact += acc["accuracy_exact"]
        total_acc_adjacent += acc["accuracy_adjacent"]

    return {
        "avg_accuracy_exact": total_acc_exact / num_samples if num_samples > 0 else 0.0,
        "avg_accuracy_adjacent": total_acc_adjacent / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_pv_task_accuracy(predictions, sample):
    """Placeholder for potential future task accuracy calculation"""
    base_round = len(predictions) // 2
    
    # use extract_number to get the first number from model response and ground truth
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = extract_letter(chat_round["value"])
            ground_truths.append(answer)
            
    # match predictions with ground truths
    preds = [extract_letter(p) for p in predictions]
    acc = sum([p == gt for p, gt in zip(preds, ground_truths)]) / len(ground_truths) if ground_truths else 0.0
    
    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1
    
    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }

def sum_pv_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]
    
    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_colorsize_task_accuracy(predictions, sample):
    base_round = len(predictions) // 2
    
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = extract_number(chat_round["value"])
            ground_truths.append(answer)
            
    # match predictions with ground truths
    preds = [extract_number(p) for p in predictions]
    acc = sum([p == gt for p, gt in zip(preds, ground_truths)]) / len(ground_truths) if ground_truths else 0.0
    
    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1
    
    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }

def sum_colorsize_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]
    
    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_leftright_task_accuracy(predictions, sample):
    base_round = len(predictions) // 2
    
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            # answer = extract_number(chat_round["value"])
            answer = extract_letter(chat_round["value"])
            ground_truths.append(answer)
            
    # match predictions with ground truths
    # preds = [extract_number(p) for p in predictions]
    preds = [extract_letter(p) for p in predictions]
    acc = sum([p == gt for p, gt in zip(preds, ground_truths)]) / len(ground_truths) if ground_truths else 0.0
    
    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1
    
    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }

def sum_leftright_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]
    
    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_spatialdetails_task_accuracy(predictions, sample):
    base_round = len(predictions) // 2
    
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            # answer = extract_number(chat_round["value"])
            answer = extract_letter(chat_round["value"])
            ground_truths.append(answer)
            
    # match predictions with ground truths
    # preds = [extract_number(p) for p in predictions]
    preds = [extract_letter(p) for p in predictions]
    acc = sum([p == gt for p, gt in zip(preds, ground_truths)]) / len(ground_truths) if ground_truths else 0.0
    
    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1
    
    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }

def sum_spatialdetails_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]
    
    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_localize_task_accuracy(predictions, sample):
    base_round = len(predictions) // 2
    
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = chat_round["value"].strip() if chat_round["value"].strip() else ""
            # ground_truths.append(extract_number(answer))
            answer = extract_letter(answer)
            ground_truths.append(answer)
            
    # match exactly
    predictions = [extract_letter(p) for p in predictions]
    accs = [p == gt for p, gt in zip(predictions, ground_truths)]
    acc = sum(accs) / len(accs) if accs else 0.

    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1

    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }

def sum_localize_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]

    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_point_count_task_accuracy(predictions, sample):
    base_round = len(predictions) // 2
    
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = extract_number(chat_round["value"])
            ground_truths.append(answer)
            
    # match predictions with ground truths
    preds = [extract_number(p) for p in predictions]
    acc = sum([p == gt for p, gt in zip(preds, ground_truths)]) / len(ground_truths) if ground_truths else 0.0
    
    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1
    
    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }
    
def sum_point_count_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]
    
    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def calc_caption_task_accuracy(predictions, sample):
    """Placeholder for potential future task accuracy calculation"""
    base_round = len(predictions) // 2
    
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = chat_round["value"].strip() if chat_round["value"].strip() else ""
            ground_truths.append(answer)
            
    # match prediction words with ground truths
    preds = [p.strip() for p in predictions]
    acc = sum([p == gt for p, gt in zip(preds, ground_truths)]) / len(ground_truths) if ground_truths else 0.0

    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1

    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }

def sum_caption_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]
    
    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }


def calc_compare_task_accuracy(predictions, sample):
    """Placeholder for potential future task accuracy calculation"""
    base_round = len(predictions) // 2
    
    ground_truths = []
    for _, chat_round in enumerate(sample["conversations"]):
        if chat_round["from"] == "gpt":
            answer = chat_round["value"].strip() if chat_round["value"].strip() else ""
            # answer = extract_number(answer)
            answer = extract_letter(answer)
            ground_truths.append(answer)
            
    # match prediction words with ground truths
    preds = [p.strip() for p in predictions]
    # preds = [extract_number(p) for p in preds]
    preds = [extract_letter(p) for p in preds]
    acc = sum([p == gt for p, gt in zip(preds, ground_truths)]) / len(ground_truths) if ground_truths else 0.0

    # put responses in sample's every gpt round
    chats = sample["conversations"].copy()
    for i, resp in enumerate(predictions):
        gpt_round_idx = 0
        for j, chat_round in enumerate(sample["conversations"]):
            if chat_round["from"] == "gpt":
                if i == gpt_round_idx:
                    chats[j]["model_response"] = resp
                    break
                gpt_round_idx += 1

    return {
        "accuracy": acc,
        "total_rounds": len(predictions),
        "base_round": base_round,
        "responses": predictions,
        "ground_truths": ground_truths,
        "image": sample["image"],
        "chats": chats
    }

def sum_compare_task_accuracies(accuracies_list):
    total_acc = 0.0
    valid_list = [acc for acc in accuracies_list if "error" not in acc]
    num_samples = len(valid_list)

    for acc in valid_list:
        total_acc += acc["accuracy"]
    
    return {
        "avg_accuracy": total_acc / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

def count_statistics(count_file):
    count_results = json.load(open(count_file, "r"))["detailed_accuracies"]

    # analyze count results list of dict
    # get every number, from 0-10, the accuracy of the model
    stats = {str(i): {"correct": 0, "total": 0} for i in range(11)}

    for res in count_results:
        if "error" not in res:
            num = str(res["ground_truths"][0])
            if res["accuracy"] == 1.0:
                stats[num]["correct"] += 1
            stats[num]["total"] += 1

    # draw bar chart
    for num, correct in stats.items():
        plt.bar(num, correct["correct"] / correct["total"])

    model_name = count_file.split("/")[-1].split("_test_results_")[0]
    plt.xlabel("Number")
    plt.ylabel("Accuracy")
    plt.title(f"Count Task Accuracy of {model_name}")
    plt.ylim(0, 1)
    plt.savefig(f"figures/{model_name}_count_task_accuracy_by_number.png")
    
def count_statistics_bar(count_files, model_names):
    """
    Args:
        count_files (list[str]): 多个 count task 结果文件路径
        model_names (list[str]): 对应的模型名称（用于图例）
    """
    plt.figure(figsize=(8, 4), dpi=300)

    n_models = len(count_files)
    numbers = np.arange(1, 13)
    bar_width = 0.8 / n_models  # 每个数字下两根柱并排

    # 🎨 高对比配色（蓝 vs 橙）
    colors = ["#89B0EE", '#F4A896']
    hatches = ["//", "-"]

    # 遍历模型文件
    for idx, (count_file, model_name) in enumerate(zip(count_files, model_names)):
        count_results = json.load(open(count_file, "r"))["detailed_accuracies"]

        # 初始化统计
        stats = {str(i): {"correct": 0, "total": 0} for i in range(1, 13)}

        for res in count_results:
            if "error" not in res:
                num = str(res["ground_truths"][0])
                if res["accuracy"] == 1.0:
                    stats[num]["correct"] += 1
                stats[num]["total"] += 1

        # 计算 accuracy
        accuracies = []
        for i in range(1, 13):
            total = stats[str(i)]["total"]
            acc = stats[str(i)]["correct"] / total if total > 0 else np.nan
            accuracies.append(acc)

        # 绘制 bar
        plt.bar(
            numbers + idx * bar_width - (n_models - 1) * bar_width / 2,
            accuracies,
            width=bar_width,
            color=colors[idx % len(colors)],
            edgecolor='black',
            linewidth=0.6,
            hatch=hatches[idx],
            label=model_name
        )

    # 图形样式
    plt.xlabel("Number", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    # plt.title("Count Task Accuracy by Number", fontsize=13)
    plt.ylim(0, 1.05)
    plt.xticks(numbers)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.2)

    # 图例
    plt.legend(frameon=False, fontsize=15, ncol=len(model_names))
    plt.tight_layout()
    plt.savefig("figures/count_task_accuracy_bar_comparison.png", bbox_inches='tight')


def re_calc_spatialdetails_task_accuracy(file):
    """Re-calculate spatialdetails task accuracy for a given sample"""
    # read json file
    data = json.load(open(file, "r"))
    detailed_accuracies = data["detailed_accuracies"]
    new_detailed_accuracies = []
    for res in detailed_accuracies:
        if extract_letter(res["chats"][1]["value"]) == extract_letter(res["chats"][1]["model_response"]):
            res["accuracy"] = 1.0
            res["ground_truths"] = [extract_letter(res["chats"][1]["value"])]
        else:
            res["accuracy"] = 0.0
            res["ground_truths"] = [extract_letter(res["chats"][1]["value"])]
        new_detailed_accuracies.append(res)
    data["detailed_accuracies"] = new_detailed_accuracies
    data["summary"] = sum_spatialdetails_task_accuracies(new_detailed_accuracies)["avg_accuracy"]
    # save json file
    json.dump(data, open(file.replace(".json", "_recalc.json"), "w"), indent=4)

if __name__ == "__main__":
    # extract /projectnb/ivc-ml/wqwang/Codelab/babydata/object_counting/count_instructions_test.json num to spell word
    # count_statistics("/home/azureuser/babydata/baby_results/count/babyllava_vit_tinyllama_count_lr1e-4_epoch20_20251023_035117/checkpoint-137_test_results_20251023-082053.json")
    # re_calc_spatialdetails_task_accuracy("/home/azureuser/babydata/baby_results/spatialdetails/babyllava_vit_tinyllama_spatialdetails_lr1e-4_epoch20_20251023_142616/checkpoint-152_test_results_20251027-092755.json")
    
    count_statistics_bar(
        [
            "/projectnb/ivc-ml/wqwang/Codelab/babydata/test_results/count/gpt-4o_test_results_20251106-155821.json",
            # "/projectnb/ivc-ml/wqwang/Codelab/babydata/test_results/count/gemini-2.5-pro_test_results_20251105-224816.json",
            # "/projectnb/ivc-ml/wqwang/Codelab/babydata/test_results/count/Qwen2.5-VL-7B-Instruct_test_results_20251106-115600.json",
            "/projectnb/ivc-ml/wqwang/Codelab/babydata/baby_results/count/babyllava_vit_tinyllama_1.0_sftmix_sft_lr5e-5_epoch5_20251110_154322/checkpoint-4836_test_results_20251112-135922.json"
        ],
        [
            "GPT-4o",
            # "Gemini-2.5-Pro",
            # "Qwen2.5-VL-7B",
            "BabyLLaVA-V2",
            
        ]
    )