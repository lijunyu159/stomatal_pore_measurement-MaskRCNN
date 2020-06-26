import time as t

class MyTimer():
    def __init__(self):
        self.unit = ['年', '月', '天', '小时', '分钟', '秒']
    # 开始计时
    def start(self):
        self.start = t.localtime()
        print('计时开始...')

    # 停止计时
    def stop(self):
        self.stop = t.localtime()
        self._calc()
        print('计时结束！')

    # 内部方法，计算运行时间
    def _calc(self):
        self.lasted = []
        self.prompt = '总共运行了'
        for index in range(6):
            self.lasted.append(self.stop[index] - self.start[index])
            if self.lasted[index]:
                self.prompt += (str(self.lasted[index]) + self.unit[index])
        print(self.prompt)
        # 初始化
        self.start = 0
        self.stop = 0
