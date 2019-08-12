f_in = open ('D://glove.6B//glove.6B.300d.txt','r',encoding='utf-8')
f_out = open ('D://glove.6B//mapping.300d.txt','w',encoding='utf-8')
cnt = 0
for row in f_in:
    row = row.split(' ')
    print (row[0])
    f_out.write(row[0]+' '+str(cnt)+'\n')
    cnt+=1