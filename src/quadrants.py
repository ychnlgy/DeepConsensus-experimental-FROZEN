import statistics

DC = [0.9953999811410904, 0.9956999826431274, 0.9958999842405319]
CN = [0.9954000043869019, 0.9951000046730042, 0.9947000050544739]
for data in [DC, CN]:
    print(statistics.mean(data), statistics.stdev(data))


