/usr/bin/awk ' 
    BEGIN { 
        FS="\n" ; RS="" 
        } 
    { 
            print "<<<SOR>>>\n", NF , $0 , "\n<<<EOR>>>\n" 
    } ' -
