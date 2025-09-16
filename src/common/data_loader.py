import os


class DataLoader:
    def __init__(self, data_dir):
        """
        初始化数据加载器。

        :param data_dir: 相对项目根目录的资源路径，例如 "resource/"
        """
        # 获取项目根路径
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        abs_data_dir = os.path.join(project_root, data_dir)

        self.data_dir = abs_data_dir
        self.data = []
        # self.index = 0
        self.load_data()

    def __iter__(self):
        return iter(self.data)
        # self.index = 0
        # return self

    def __len__(self):
        return len(self.data)

    # def __next__(self):
    #     if self.index < len(self.data):
    #         item = self.data[self.index]
    #         self.index += 1
    #         return item
    #     else:
    #         raise StopIteration

    def load_data(self):
        """
        加载数据目录下的所有文件完整路径。
        """
        if os.path.isdir(self.data_dir):
            self.data = sorted(
                [
                    os.path.join(self.data_dir, filename)
                    for filename in os.listdir(self.data_dir)
                    if os.path.isfile(os.path.join(self.data_dir, filename))
                ]
            )
        else:
            raise ValueError(f"{self.data_dir} 不是一个有效的目录")
