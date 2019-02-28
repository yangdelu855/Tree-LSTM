# -*- coding: utf-8 -*-

import json
import os


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        self.__name = name
        self.PADDING = "</pad>"  # padding label
        self.UNKNOWN = "</unk>"  # out of vacab label
        self.label = label  # if label alphabet
        self.instance2index = {}
        self.instances = []  # if not label,[</pad>,</unk>,......]
        self.keep_growing = keep_growing

        self.next_index = 0  # start from index 0
        if not self.label:
            self.add(self.PADDING)  # default index 0
            self.add(self.UNKNOWN)  # default index 1

    def clear(self, keep_growing=True):
        """
        reset
        :param keep_growing:
        :return:
        """
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.next_index = 0

    def add(self, instance):
        """
        add one instance，and make sure of id
        :param instance:
        :return:
        """
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        """
        get index
        :param instance:
        :return:
        """
        try:
            return self.instance2index[instance]
        except KeyError:
            #if self.keep_growing:
            index = self.next_index
            self.add(instance)
            return index
            '''else:
                return self.instance2index[self.UNKNOWN]'''  # index 1 (label will not be in this case)

    def get_instance(self, index):
        """
        get instance
        :param index:
        :return:
        """
        try:
            return self.instances[index]
        except IndexError:
            return self.instances[1]  # unknown </unk>

    def size(self):
        return len(self.instances)

    def iteritems(self):
        return self.instance2index.iteritems()

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
