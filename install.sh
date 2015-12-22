#/bin/sh

incdir="/usr/local/include/numcp"

if [ ! -e ${incdir} ]; then
	sudo mkdir ${incdir}
fi

cp ./include/*.hpp ${incdir}
cp ./include/*.h ${incdir}
echo "Installation done."

