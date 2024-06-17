import shutil

def tikz_boxplot(input_file, name, n_box):
    try:
        f = open(name, 'r')
        if f.readline()=='\\begin{tikzpicture}':
            shutil.copyfile('input.txt',name)
    except:
        shutil.copyfile('input.txt',name)
    f_w = open(name, 'a')
    boxplot = []
    for i in range(n_box):
        boxplot.append({'median':0,
                    'upper quartile':0,
                    'lower quartile':0,
                    'upper whisker':0,
                    'lower whisker':0})
    with open(input_file, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('table'):
                line = f.readline()
                l = line.split(' ')
                if len(l[0])==1:
                    boxplot[int(l[0])]['lower quartile'] = float(l[1])
                    line = f.readline()
                    l = line.split(' ')
                    boxplot[int(l[0])]['lower whisker'] = float(l[1])
                    for i in range(4):
                        line = f.readline()
                    l = line.split(' ')
                    boxplot[int(l[0])]['upper quartile'] = float(l[1])
                    line = f.readline()
                    l = line.split(' ')
                    boxplot[int(l[0])]['upper whisker'] = float(l[1])
                elif l[0]=='-0.4':
                    for i in range(n_box):
                        boxplot[i]['median'] = float(l[1])
                        for i in range(4):
                            line = f.readline()
                        l = f.readline().split(' ')
            line = f.readline()   
    f.close()
    for i in range(n_box):
        f_w.write('\n\\addplot+[\n')
        f_w.write('fill, draw=black,\n')
        f_w.write('boxplot prepared={\n')
        keys = boxplot[i].keys()
        for k in keys:
            f_w.write('\t'+k+'='+str(boxplot[i][k])+',\n')
        f_w.write('\tdraw position='+str(i+1)+'\n')
        f_w.write('},\n')
        f_w.write('] coordinates {};\n')
    f_w.write('\\end{axis}\n')
    f_w.write('\\end{tikzpicture}')
    f_w.close()

tikz_boxplot('cir_estimation_sim/plot/var_snrfc_60a_5.tex','cir_estimation_sim/plot/var_snr.tex',5)