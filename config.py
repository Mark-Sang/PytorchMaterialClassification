import warnings

class DefaultConfig():
    env = "default"                                        #visdom环境
    vis_port = 8097                                        #visdom端口
    train_iron_front_root = "./train/iron/front"           #训练 铁正视图
    train_iron_lateral_root = "./train/iron/lateral"       #训练 铁侧视图
    train_wood_front_root = "./train/wood/front"           #训练 木头正视图
    train_wood_lateral_root = "./train/wood/lateral"       #训练 木头侧视图
    test_front_root = "./test/front"                       #测试 正视图
    test_lateral_root = "./test/lateral"                   #测试 侧视图
    use_gpu = True                                         #使用GPU加速
    result_file = "./testresult.csv"                       #test测试集结果

    def parse(self, kwargs):                           #根据字典更新config参数，便于命令行更改
        for k,v in kwargs.items():                     #更新配置参数
            if not hasattr(self, k):
                warnings.warn("Warning:设置文件里没有这个参数")
            setattr(self, k, v)

        print("user config:")
        for k,v in self.__class__.__dict__.items():
            if (not k.startswith('__') and not k.startswith('p')):                
                print(k,getattr(self, k))
        print('\n''\n')