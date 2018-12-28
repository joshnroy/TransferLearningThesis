import numpy as np

def calc(Hin, Win, padding, dilation, kernel_size, stride):
    Hout = (Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0] + 1
    Wout = (Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1] + 1
    return Hout, Wout

def main():
    padding = []
    dilation = []
    kernel_size = []
    stride = []
    print("What is Hin?")
    Hin = int(input("Hin: "))
    print("What is Win?")
    Win = int(input("Win: "))
    print("What is padding y?")
    padding.append(int(input("Paddingy: ")))
    print("What is padding x?")
    padding.append(int(input("Paddingx: ")))
    print("What is dilation y?")
    dilation.append(int(input("dilationy: ")))
    print("What is dilation x?")
    dilation.append(int(input("dilationx: ")))
    print("What is kernel_size y?")
    kernel_size.append(int(input("kernel_sizey: ")))
    print("What is kernel_size x?")
    kernel_size.append(int(input("kernel_sizex: ")))
    print("What is stride y?")
    stride.append(int(input("stridey: ")))
    print("What is stride x?")
    stride.append(int(input("stridex: ")))
    
    Hout, Wout = calc(Hin, Win, padding, dilation, kernel_size, stride)
    print("The new dimensions are (" + str(Hout) + ", " + str(Wout) + ").")

if __name__ == "__main__":
    main()