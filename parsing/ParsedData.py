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

    def get(self):
        return self.data

    def get_by_index(self, i):
        return self.data[i]

    def filter_by_digit(self, d):
        return ParsedData([data_block for data_block in self.data if data_block.digit == d])

