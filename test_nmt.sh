

## system
#./LongTest -m system.model.base.6-4.bin  -v vocab.system.nn -b 2 -g 2 -s c2e-system -i 03.seg.bpe -o 03.system.addtop200.out
#./LongTest -m system.model.base.6-4.bin  -v vocab.system.nn -b 2 -g 2 -i 03.seg.bpe -o 03.system.nosmall.out
#./LongTest -m ./models-old/system.model.base.6-4.bin  -v ./models-old/vocab.system.nn -b 4 -g 1 -i 03.seg.bpe -o 03.system.std.out
#./LongTest -m ./models-old/system.model.base.6-4.bin  -v ./models-old/vocab.system.nn -b 2 -g 1 -s ./models-old/c2e-system -i 03.seg.bpe -o 03.system.b1.out

./LongTest -m ./models/model.base.6-6.bin  -v ./models/vocab.nn -b 4 -g 1 -i 03.seg.bpe -o 03.system.std.out
