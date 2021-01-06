import os

def main():
	#subprocess.call(['java', '-jar', '/home/valia/Desktop/CPPred-RF_local/CPPred-RF_local.jar input.txt result.txt'])
	os.system("java -jar CPPred-RF_local.jar input.txt result.txt")

if __name__ == '__main__':
    main()
