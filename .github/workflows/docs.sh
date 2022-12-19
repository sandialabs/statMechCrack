#!/bin/bash

shopt -s extglob
cd examples/ && cp -r !(statmechcrack) ../docs/ && cd ../
mkdir docs-temp/
mv docs/* docs-temp/
cd docs/
export VERSION=$(grep __version__ ../statmechcrack/__init__.py | cut -f2 -d '"')
sphinx-quickstart --sep -p statmechcrack -l en -a 'Michael R. Buche, Scott J. Grutzik' -r ${VERSION} -v ${VERSION}
sphinx-apidoc -e -P -o source ../ ../*setup*
mv ../docs-temp/* source/
rm -r ../docs-temp/
for file in ../statmechcrack/*.py; do
    export file_basename=$(basename ${file%.*})
    export rst_file="source/*$(basename ${file%.*}).rst"
    if [ -f $rst_file ]; then
        if grep -q :cite $file; then
            echo "citations in $file"
            echo "" >> $rst_file
            echo "" >> $rst_file
            export OLD_CITE="cite:\`"
            export NEW_CITE="${OLD_CITE}${file_basename}-"
            sed -i -e "s/${OLD_CITE}/${NEW_CITE}/g" $file
            echo ".. raw::" >> $rst_file
            echo " html" >> $rst_file
            echo "" >> $rst_file
            echo -n "   <hr>" >> $rst_file
            echo "" >> $rst_file
            echo "" >> $rst_file
            echo "**References**" >> $rst_file
            echo "" >> $rst_file
            echo ".. bibliography::" >> $rst_file
            echo -n "   :filter:" >> $rst_file
            echo " docname in docnames" >> $rst_file
            echo -n "   :keyprefix:" >> $rst_file
            echo " $file_basename-" >> $rst_file
        fi
    fi
done
