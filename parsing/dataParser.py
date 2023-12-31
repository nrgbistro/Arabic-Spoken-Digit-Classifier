from parsing.DataBlock import DataBlock


def parse_file(file_name, num_speakers):
    ret = []  # [(digit, gender, token_index, ['MFCC1', 'MFCC2']), ...]

    with open(file_name, "r") as file:
        i = 1
        speaker = "M"
        digit = 0
        digit_index = 1
        current_mfccs = []
        gender_cutoff = num_speakers * 5
        digit_cutoff = num_speakers * 10
        for line in file:
            if len(line.strip()) == 0 and len(current_mfccs) > 0:
                ret.append(DataBlock(digit, digit_index, speaker, current_mfccs))
                if i % gender_cutoff == 0:
                    speaker = "F" if speaker == "M" else "M"
                if i % digit_cutoff == 0:
                    digit += 1
                if digit_index == 10:
                    digit_index = 0
                i += 1
                digit_index += 1
                current_mfccs = []
            elif len(line.strip()) == 0:
                continue
            else:
                current_mfccs.append([float(i) for i in line.strip().split(" ")])
        ret.append(DataBlock(digit, digit_index, speaker, current_mfccs))  # last one
    return ret
