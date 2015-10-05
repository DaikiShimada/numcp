#/bin/sh

incdir="/usr/local/include/numcp"

if [ ! -e ${incdir} ]; then
	sudo mkdir ${incdir}
fi

cp ./include/numcp/*.hpp ${incdir}
echo "Installation done."
