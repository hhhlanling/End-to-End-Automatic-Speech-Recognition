#import的功能的是调用了其他的库
import tensorflow as tf
from tensorflow.keras import layers


def get_layers(context = 300, filt_length_conv_1 = 150, filt_length_conv_2 = 5, 
              number_filters_conv_1 = 100, number_filters_conv_2 = 80, fc_cells = 512):
    '''
    Python 使用三个连续的单引号注释多行内容
    这里说明了函数的功能、结构和参考文献
    Returns the paramters that for a CNN based model.
    Consider chaning default parameters as per application.
    
    Architecture: CNN->FC->Logits
    
    Reference:
    "Analysis of CNN-based Speech Recognition System using Raw Speech as Input"
    (https://ronan.collobert.com/pub/matos/2015_cnnspeech_interspeech.pdf)
    到这里也都还是注释
    '''

    #tf.keras.layers.Input()用于构建网络的第一层——输入层
    #参数shape：输入的形状，tuple(元祖)类型。不含batch_size；tuple的元素可以为None类型数据，表示未知的或者说任意的，一般这里不用None 
                #在Python中，元组（Tuple）是一种不可变的序列类型，它在功能上类似于列表（List），
                #但有一个关键的区别：一旦创建，元组中的元素就不能被修改。
                #这种不可变性使得元组在某些情况下比列表更适用，特别是在需要保证数据不被改变的场景中。
    #参数name：给layers起个名字，在整个网络中不能出现重名。如果name=None，则系统会自动为该层创建名字             
    y_true = tf.keras.layers.Input((None,), name="y_true")
    y_true_length = tf.keras.layers.Input((1), name="y_true_length")

    # Inputshape [btch_sz, n_time, n_channels = 1]
    input_audio = layers.Input(shape=(None, 1), name="audio_input")

    # Append zeros in time for context at the beggining and end of audio sample 在音频样本的开始和结束处及时添加零
    #“ZeroPadding1D”的功能是对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度
    #padding：整数，表示在要填充的轴的起始和结束处填充0的数目。从上文可知context是300
    #(input_audio)就是指这个函数的输入变量，至于name = 'zero_padding_context'，也没出现过第二次，不知道干什么用的   
    input_audio_padded = layers.ZeroPadding1D(padding=(context), name = 'zero_padding_context')(input_audio)



    '''
    Conv1D 
    1D 卷积层 (例如时序卷积)。
    该层创建了一个卷积核，该卷积核对层输入进行卷积， 以生成输出张量。 
    如果 use_bias 为 True， 则会创建一个偏置向量并将其添加到输出中。 
    最后，如果 activation 不是 None，它也会应用于输出。
    filter：输出空间的维度 （即卷积中滤波器的输出数量）。
    Kernel_size：指定卷积窗口的大小。
    Strides：指定卷积的步幅长度。
    '''
    # Apply 1d filter: Shape [btch_sz, n_time, n_channels]
    filt_length_conv_1 = 2 * context + 1 + filt_length_conv_1
    conv = layers.Conv1D(number_filters_conv_1, filt_length_conv_1, strides=100, activation='relu', name="conv_1")(input_audio_padded)
    conv = layers.Conv1D(number_filters_conv_2, filt_length_conv_2, strides = 1, activation='relu', name="conv_2")(conv)
    conv = layers.Conv1D(number_filters_conv_2, filt_length_conv_2, strides = 1, activation='relu', name="conv_3")(conv)

    # Apply FC layer to each time step 正式代码中，fc作为变量在这里第一次出现。python里面不需要定义变量，直接就用
    fc = layers.TimeDistributed(layers.Dense(fc_cells), name = 'fc')(conv)
    fc = layers.ReLU()(fc)
    fc = layers.Dropout(rate=0.1)(fc)

    # Apply FC layer to each time step to output prob distro across chars
    logits = layers.TimeDistributed(layers.Dense(29, activation='softmax'), name = 'logits')(fc)

    return logits, input_audio, y_true, y_true_length

