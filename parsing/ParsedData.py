from dataclasses import dataclass


def get_all_blocks(d):
    ret = []
    for data_block in d:
        for mfcc in data_block.mfccs:
            ret.append(mfcc)
    return ret


@dataclass(frozen=True, order=True)
class ParsedData:
    data: list

    def __iter__(self):
        for item in self.data:
            yield item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __str__(self):
        ret = ""
        for data_block in self.data:
            ret += str(data_block) + "\n"
        return ret

    def shape(self):
        return len(self.data), len(self.data[0].mfccs), len(self.data[0].mfccs[0])

    def get(self):
        return self.data

    def get_by_index(self, i):
        return self.data[i]

    def filter_by_digit(self, d):
        return ParsedData([data_block for data_block in self.data if data_block.digit == d])

    def filter_by_mfccs(self, mfcc_indexes):
        new_data = []
        for data_block in self.data:
            new_data.append(data_block.filter_mfccs(mfcc_indexes))
        return ParsedData(new_data)

