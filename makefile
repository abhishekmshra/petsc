# $Id: makefile,v 1.296 1999/12/22 04:03:55 bsmith Exp bsmith $ 
#
# This is the makefile for installing PETSc. See the file
# docs/installation.html for directions on installing PETSc.
# See also bmake/common for additional commands.
#
ALL: all

DIRS	   = src include docs 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

#
# Basic targets to build PETSc libraries.
# all     : builds the c, fortran, and f90 libraries
all       : info info_h chkpetsc_dir deletelibs build_c build_fortran shared
#
# Prints information about the system and version of PETSc being compiled
#
info:
	-@echo "=========================================="
	-@echo " "
	-@echo "See docs/troubleshooting.html and docs/bugreporting.html"
	-@echo "for help with installation problems. Please send EVERYTHING"
	-@echo "printed out below when reporting problems"
	-@echo " "
	-@echo "To subscribe to the PETSc users mailing list, send mail to "
	-@echo "majordomo@mcs.anl.gov with the message: "
	-@echo "subscribe petsc-users"
	-@echo " "
	-@echo "=========================================="
	-@echo On `date` on `hostname`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS}"
	-@if [ -n "${C_CCV}" -a "${C_CCV}" != "unknown" ] ; then \
	  echo "C Compiler version:" ; ${C_CCV} ; fi
	-@if [ -n "${CXX_CCV}" -a "${CXX_CCV}" != "unknown" ] ; then \
	  echo "C++ Compiler version:" ; ${CXX_CCV} ; fi
	-@echo "Using Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}"
	-@if [ -n "${C_FCV}" -a "${C_FCV}" != "unknown" ] ; then \
	  echo "Fortran Compiler version:" ; ${C_FCV} ; fi
	-@echo "-----------------------------------------"
	-@grep PETSC_VERSION_NUMBER include/petsc.h | sed "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags:"
	-@grep "define " bmake/${PETSC_ARCH}/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${PETSC_INCLUDE}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "------------------------------------------"
	-@echo "Using C linker: ${CLINKER}"
	-@echo "Using Fortran linker: ${FLINKER}"
	-@echo "Using libraries: ${PETSC_LIB}"
	-@echo "=========================================="
#
#
MINFO = ${PETSC_DIR}/bmake/${PETSC_ARCH}/petscmachineinfo.h
info_h:
	-@$(RM) -f MINFO
	-@echo  "static char *petscmachineinfo = \"  " >> MINFO
	-@echo  "Libraries compiled on `date` on `hostname` " >> MINFO
	-@echo  Machine characteristics: `uname -a` "" >> MINFO
	-@echo  "-----------------------------------------" >> MINFO
	-@echo  "Using C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS} " >> MINFO
	-@if [  "${C_CCV}" -a "${C_CCV}" != "unknown" ] ; then \
	  echo  "C Compiler version:"  >> MINFO ; ${C_CCV} >> MINFO 2>&1; fi
	-@if [  "${CXX_CCV}" -a "${CXX_CCV}" != "unknown" ] ; then \
	  echo  "C++ Compiler version:"  >> MINFO; ${CXX_CCV} >> MINFO 2>&1 ; fi
	-@echo  "Using Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}" >> MINFO
	-@if [  "${C_FCV}" -a "${C_FCV}" != "unknown" ] ; then \
	  echo  "Fortran Compiler version:" >> MINFO ; ${C_FCV} >> MINFO 2>&1 ; fi
	-@echo  "-----------------------------------------" >> MINFO
	-@echo  "Using PETSc flags: ${PETSCFLAGS} ${PCONF}" >> MINFO
	-@echo  "-----------------------------------------" >> MINFO
	-@echo  "Using configuration flags:" >> MINFO
	-@echo  "-----------------------------------------" >> MINFO
	-@echo  "Using include paths: ${PETSC_INCLUDE}" >> MINFO
	-@echo  "-----------------------------------------" >> MINFO
	-@echo  "Using PETSc directory: ${PETSC_DIR}" >> MINFO
	-@echo  "Using PETSc arch: ${PETSC_ARCH}" >> MINFO
	-@echo  "------------------------------------------" >> MINFO
	-@echo  "Using C linker: ${CLINKER}" >> MINFO
	-@echo  "Using Fortran linker: ${FLINKER}" >> MINFO
	-@cat MINFO | sed -e 's/$$/  \\n\\/' > ${MINFO}
	-@echo  "Using libraries: ${PETSC_LIB} \"; " >> ${MINFO}
	$(RM) MINFO
#
# Builds the PETSc libraries
# This target also builds fortran77 and f90 interface
# files. (except compiling *.F files)
#
build_c:
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=libfast tree 
	-@cd ${PETSC_DIR}/src/sys/src/time ; \
	${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} rs6000_asmtime
	${RANLIB} ${PDIR}/*.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed building libraries"
	-@echo "========================================="

#
# Builds PETSc Fortran source
# Note:	 libfast cannot run on .F files on certain machines, so we
# use libf to compile the fortran source files.
#
build_fortran:
	-@echo "BEGINNING TO COMPILE FORTRAN SOURCE"
	-@echo "========================================="
	-@cd src/fortran/custom; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} libf clean 
	-@cd src/fortran/kernels; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} libf clean
	${RANLIB} ${PDIR}/libpetscfortran.a
	${RANLIB} ${PDIR}/libpetsc.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed compiling Fortran source"
	-@echo "========================================="

petscblas: info chkpetsc_dir
	-${RM} -f ${PDIR}/libpetscblas.*
	-@echo "BEGINNING TO COMPILE C VERSION OF BLAS AND LAPACK"
	-@echo "========================================="
	-@cd src/blaslapack; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=libfast tree
	${RANLIB} ${PDIR}/libpetscblas.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed compiling C version of BLAS and LAPACK"
	-@echo "========================================="


# Builds PETSc test examples for a given BOPT and architecture
testexamples: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_1  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "========================================="

# Builds PETSc test examples for a given BOPT and architecture
testfortran: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN FORTRAN TEST EXAMPLES"
	-@echo "========================================="
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines or the way Fortran formats numbers"
	-@echo "some of the results may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_3  tree 
	-@echo "Completed compiling and running Fortran test examples"
	-@echo "========================================="

# Builds PETSc test examples for a given BOPT and architecture
testexamples_uni: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_4  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="
testfortran_uni: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR FORTRAN EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_9  tree 
	-@echo "Completed compiling and running uniprocessor fortran test examples"
	-@echo "========================================="
matlabcodes:
	-@echo "BEGINNING TO COMPILE MATLAB INTERFACE"
	-@cd src/sys/src/viewer/impls/socket/matlab; ${OMAKE} BOPT=g matlabcodes  PETSC_ARCH=${PETSC_ARCH}

# Ranlib on the libraries
ranlib:
	${RANLIB} ${PDIR}/*.a

# Deletes PETSc libraries
deletelibs: chkopts_basic
	-${RM} -f ${PDIR}/*


# ------------------------------------------------------------------
#
# All remaining actions are intended for PETSc developers only.
# PETSc users should not generally need to use these commands.
#

# To access the tags in EMACS, type M-x visit-tags-table and specify
# the file petsc/TAGS.	
# 1) To move to where a PETSc function is defined, enter M-. and the
#     function name.
# 2) To search for a string and move to the first occurrence,
#     use M-x tags-search and the string.
#     To locate later occurrences, use M-,
# Builds all etags files
alletags:
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSDIR=${PETSC_DIR} etags
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSDIR=${PETSC_DIR} etags_complete
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSDIR=${PETSC_DIR} etags_noexamples
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSDIR=${PETSC_DIR} etags_examples
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSDIR=${PETSC_DIR} etags_makefiles
# Builds the basic etags file.	This should be employed by most users.
etags:
	-${RM} ${TAGSDIR}/TAGS
	-touch ${TAGSDIR}/TAGS
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS ACTION=etags_sourcec alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS ACTION=etags_sourceh alltree
	-cd src/fortran; ${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS ACTION=etags_sourcef alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS ACTION=etags_examplesc alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS ACTION=etags_examplesf alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS ACTION=etags_examplesch alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS ACTION=etags_examplesfh alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS ACTION=etags_makefile alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS etags_bmakefiles
	-chmod g+w TAGS
# Builds complete etags list; only for PETSc developers.
etags_complete:
	-${RM} ${TAGSDIR}/TAGS_COMPLETE
	-touch ${TAGSDIR}/TAGS_COMPLETE
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_sourcec alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_sourceh alltree
	-cd src/fortran; ${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_sourcef alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_examplesc alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_examplesf alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_examplesch alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_examplesfh alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_makefile alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE etags_bmakefiles
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_docs alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_COMPLETE ACTION=etags_scripts alltree
	-chmod g+w TAGS_COMPLETE
# Builds the etags file that excludes the examples directories
etags_noexamples:
	-${RM} ${TAGSDIR}/TAGS_NO_EXAMPLES
	-touch ${TAGSDIR}/TAGS_NO_EXAMPLES
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_NO_EXAMPLES ACTION=etags_sourcec alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_NO_EXAMPLES ACTION=etags_sourceh alltree
	-cd src/fortran; ${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_NO_EXAMPLES ACTION=etags_sourcef alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_NO_EXAMPLES ACTION=etags_makefile alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_NO_EXAMPLES etags_bmakefiles
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_NO_EXAMPLES ACTION=etags_docs alltree
	-chmod g+w TAGS_NO_EXAMPLES
# Builds the etags file for makefiles
etags_makefiles: 
	-${RM} ${TAGSDIR}/TAGS_MAKEFILES
	-touch ${TAGSDIR}/TAGS_MAKEFILES
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_MAKEFILES ACTION=etags_makefile alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_MAKEFILES etags_bmakefiles
	-chmod g+w TAGS_MAKEFILES
# Builds the etags file for examples
etags_examples: 
	-${RM} ${TAGSDIR}/TAGS_EXAMPLES
	-touch ${TAGSDIR}/TAGS_EXAMPLES
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_EXAMPLES ACTION=etags_examplesc alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_EXAMPLES ACTION=etags_examplesch alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_EXAMPLES ACTION=etags_examplesf alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_EXAMPLES ACTION=etags_examplesfh alltree
	-chmod g+w TAGS_EXAMPLES
etags_fexamples: 
	-${RM} ${TAGSDIR}/TAGS_FEXAMPLES
	-touch ${TAGSDIR}/TAGS_FEXAMPLES
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_FEXAMPLES ACTION=etags_examplesf alltree
	-${OMAKE} PETSC_DIR=${PETSC_DIR} TAGSFILE=${TAGSDIR}/TAGS_EXAMPLES ACTION=etags_examplesfh alltree
	-chmod g+w TAGS_FEXAMPLES
#
# These are here for the target allci and allco, and etags
#

BMAKEFILES = bmake/common* bmake/*/base bmake/*/base_variables bmake/*/base.site \
	     bmake/*/petscconf.h bmake/win32/makefile.dos bin/config/base*.in
DOCS	   = bmake/readme bmake/petscconf.defs
SCRIPTS    = maint/addlinks maint/builddist maint/buildlinks maint/wwwman \
	     maint/xclude maint/crontab  \
	     maint/autoftp include/foldinclude/generateincludes

# Builds all the documentation - should be done every night
alldoc: allmanpages
	cd docs/tex/manual; ${OMAKE} manual.dvi manual.ps manual.html splitmanual.html

# Deletes man pages (HTML version)
deletemanualpages:
	${RM} -f ${PETSC_DIR}/docs/manualpages/*/*.html \
                 ${PETSC_DIR}/docs/manualpages/manualpages.cit 

# Deletes man pages (LaTeX version)
deletelatexpages:
	${RM} -f ${PETSC_DIR}/docs/tex/rsum/*sum*.tex

# Builds all versions of the man pages
allmanpages: allmanualpages alllatexpages
allmanualpages: deletemanualpages
	-${OMAKE} ACTION=manualpages_buildcite ttree
	-${OMAKE} ACTION=manualpages ttree
	-maint/wwwindex.py ${PETSC_DIR}
	-maint/examplesindex.tcl
	-maint/htmlkeywords.tcl
	-@chmod g+w docs/manualpages/*/*.html

alllatexpages: deletelatexpages
	-${OMAKE} ACTION=latexpages ttree
	-@chmod g+w docs/tex/rsum/*

# Builds Fortran stub files
allfortranstubs:
	-@include/foldinclude/generateincludes
	-@${RM} -f src/fortran/auto/*.c
	-${OMAKE} ACTION=fortranstubs ttree
	-@cd src/fortran/auto; ${OMAKE} -f makefile fixfortran
	chmod g+w src/fortran/auto/*.c

allci: 
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=ci  alltree 

allco: 
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=co  alltree 

#
#   The commands below are for generating ADIC versions of the code;
# they are not currently used.
#
alladicignore:
	-@${RM} ${PDIR}/adicignore
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adicignore  tree 

alladic:
	-@echo "Beginning to compile ADIC source code in all directories"
	-@echo "Using ADIC compiler: ${ADIC_CC} ${CCPPFLAGS}"
	-@echo "========================================="
	-@cd include ; \
           ${ADIC_CC} -s -f 1 ${CCPPFLAGS} petsc.h 
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adic  tree 
	-@cd src/inline ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} adic
	-@cd src/blaslapack ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adic  tree

alladiclib:
	-@echo "Beginning to compile ADIC libraries in all directories"
	-@echo "Using compiler: ${CC} ${COPTFLAGS}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags:"
	-@grep "define " bmake/${PETSC_ARCH}/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${PETSC_INCLUDE}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "========================================="
	-@${RM} -f  ${PDIR}/*adic.a
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adiclib  tree
	-@cd src/blaslapack ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adiclib  tree
	-@cd src/adic/src ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} lib

# -------------------------------------------------------------------------------
#
# Some macros to check if the fortran interface is up-to-date.
#
countfortranfunctions: 
	-@cd ${PETSC_DIR}/src/fortran; egrep '^void' custom/*.c auto/*.c | \
	cut -d'(' -f1 | tr -s  ' ' | cut -d' ' -f2 | uniq | egrep -v "(^$$|Petsc)" | \
	sed "s/_$$//" | sort > /tmp/countfortranfunctions

countcfunctions:
	-@ grep extern ${PETSC_DIR}/include/*.h *.h | grep "(" | tr -s ' ' | \
	cut -d'(' -f1 | cut -d' ' -f3 | grep -v "\*" | tr -s '\012' |  \
	tr 'A-Z' 'a-z' |  sort > /tmp/countcfunctions

difffortranfunctions: countfortranfunctions countcfunctions
	-@echo -------------- Functions missing in the fortran interface ---------------------
	-@diff /tmp/countcfunctions /tmp/countfortranfunctions | grep "^<" | cut -d' ' -f2
	-@echo ----------------- Functions missing in the C interface ------------------------
	-@diff /tmp/countcfunctions /tmp/countfortranfunctions | grep "^>" | cut -d' ' -f2
	-@${RM}  /tmp/countcfunctions /tmp/countfortranfunctions

checkbadfortranstubs:
	-@echo "========================================="
	-@echo "Functions with MPI_Comm as an Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; grep '^void' *.c | grep 'MPI_Comm' | \
	tr -s ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3
	-@echo "========================================="
	-@echo "Functions with a String as an Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; grep '^void' *.c | grep 'char \*' | \
	tr -s ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3
	-@echo "========================================="
	-@echo "Functions with Pointers to PETSc Objects as Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; \
	_p_OBJ=`grep _p_ ${PETSC_DIR}/include/*.h | tr -s ' ' | \
	cut -d' ' -f 3 | tr -s '\012' | grep -v '{' | cut -d'*' -f1 | \
	sed "s/_p_//g" | tr -s '\012 ' ' *|' ` ; \
	for OBJ in $$_p_OBJ; do \
	grep "$$OBJ \*" *.c | tr -s ' ' | tr -s ':' ' ' | \
	cut -d'(' -f1 | cut -d' ' -f1,3; \
	done 
# Builds noise routines (not yet publically available)
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
noise: info chkpetsc_dir
	-@echo "Beginning to compile noise routines"
	-@echo "========================================="
	-@cd src/snes/interface/noise; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} lib > trashz 2>&1; \
	  grep -v clog trashz | grep -v "information sections" | \
	  egrep -i '(Error|warning|Can)' >> /dev/null;\
	  if [ "$$?" != 1 ]; then \
	  cat trashz ; fi; ${RM} trashz
	${RANLIB} ${PDIR}/libpetscsnes.a
	-@chmod g+w  ${PDIR}/libpetscsnes.a
	-@echo "Completed compiling noise routines"
	-@echo "========================================="

