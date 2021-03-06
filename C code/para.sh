echo "#define _BI 32" >bi.h
make clean
make
./benchmark_blocked >blocked_32.out
./benchmark_rb >rb_32.out
./benchmark_copy >copy_32.out
./benchmark_autovect >autovect_32.out
./benchmark_genvect >genvect_32.out


echo "#define _BI 64" >bi.h
make clean
make
./benchmark_blocked >blocked_64.out
./benchmark_rb >rb_64.out
./benchmark_copy >copy_64.out
./benchmark_autovect >autovect_64.out
./benchmark_genvect >genvect_64.out

echo "#define _BI 96" >bi.h
make clean
make
./benchmark_blocked >blocked_96.out
./benchmark_rb >rb_96.out
./benchmark_copy >copy_96.out
./benchmark_autovect >autovect_96.out
./benchmark_genvect >genvect_96.out

echo "#define _BI 128" >bi.h
make clean
make
./benchmark_blocked >blocked_128.out
./benchmark_rb >rb_128.out
./benchmark_copy >copy_128.out
./benchmark_autovect >autovect_128.out
./benchmark_genvect >genvect_128.out

echo "#define _BI 160" >bi.h
make clean
make
./benchmark_blocked >blocked_160.out
./benchmark_rb >rb_160.out
./benchmark_copy >copy_160.out
./benchmark_autovect >autovect_160.out
./benchmark_genvect >genvect_160.out

echo "#define _BI 192" >bi.h
make clean
make
./benchmark_blocked >blocked_192.out
./benchmark_rb >rb_192.out
./benchmark_copy >copy_192.out
./benchmark_autovect >autovect_192.out
./benchmark_genvect >genvect_192.out

echo "#define _BI 224" >bi.h
make clean
make
./benchmark_blocked >blocked_224.out
./benchmark_rb >rb_224.out
./benchmark_copy >copy_224.out
./benchmark_autovect >autovect_224.out
./benchmark_genvect >genvect_224.out


