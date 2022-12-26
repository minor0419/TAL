import subprocess

data = []
adr = 'https://www.crcv.ucf.edu/THUMOS14/test_set/TH14_test_set_mp4/'
f = open('./video_test.txt', 'r')
while (True):
    d = f.readline()
    if d == '':
        break
    data.append(d)
f.close()

for text in data:
    text = text.replace('\n', '')
    url = adr + text
    cmd = ['wget', '--no-check-certificate', url, '-P /home/minoru/THUMOS14/test', ]
    result = subprocess.run(cmd)
    print(result.stdout)
