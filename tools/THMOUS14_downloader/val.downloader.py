import subprocess

data = []
adr = 'https://www.crcv.ucf.edu/THUMOS14/Validation_set/videos/'
f = open('./video_validation.txt', 'r')
while (True):
    d = f.readline()
    if d == '':
        break
    data.append(d)
f.close()

for text in data:
    text = text.replace('\n', '')
    url = adr + text
    cmd = ['wget', '--no-check-certificate', url, '-P /home/minoru/THUMOS14/val', ]
    result = subprocess.run(cmd)
    print(result.stdout)
