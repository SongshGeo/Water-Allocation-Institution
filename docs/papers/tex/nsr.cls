%%
%% This is file `nsr.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% classes.dtx  (with options: `nsr')
%% 
%% This is a generated file.
%% 
%% Copyright 2021 Aptara India
%% The LaTeX3 Project and any individual authors listed elsewhere
%% in this file.
%% 
%% This file was generated from file(s) of the LaTeX base system.
%% --------------------------------------------------------------
%% 
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2003/12/01 or later.
%% 
%% This file has the LPPL maintenance status "maintained".
%% 
%% This file may only be distributed together with a copy of the LaTeX
%% base system. You may however distribute the LaTeX base system without
%% such generated files.
%% 
%% The list of all files belonging to the LaTeX base distribution is
%% given in the file `manifest.txt'. See also `legal.txt' for additional
%% information.
%% 
%% The list of derived (unpacked) files belonging to the distribution
%% and covered by LPPL is defined by the unpacking scripts (with
%% extension .ins) which are part of the distribution.
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}[2021/12/17]
\ProvidesClass{nsr}
              [2021/12/17 v1.0f
 Standard LaTeX document class]
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagefalse
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
\ExecuteOptions{letterpaper,10pt,oneside,onecolumn,final}
\ProcessOptions
%\input{size1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\def\ps@headings{%
   \def\@evenhead{\hspace*{-30mm}\hbox to \typewidth{\hfill\rhhfont{{\itshape Natl Sci Rev},\ \@jyear,\ Vol.\ \@jvol, \jobname }\hfill}}%
   \let\@oddhead\@evenhead%
   \def\@oddfoot{\hspace*{-30mm}\hbox to \typewidth{\hfill\oddfolio\hfill}}%
   \def\@evenfoot{\hspace*{-30mm}\hbox to \typewidth{\hfill\evenfolio\hfill}}%  
   \let\@mkboth\@gobbletwo}
%%
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
    }
\newcommand\maketitle{\par
  \begingroup
  \newpage
  \global\@topnum\z@   % Prevents figures from going at top of page.
  \twocolumn[\@maketitle]%
  \thispagestyle{plain}\@thanks%
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
  \newpage
  \null
 \bgroup\parindent0pt%
  \vskip 18.5pt%
  \let \footnote \thanks
  \hbox{\hspace*{-7.25pc}\vbox to 0pt{\hsize6pc\vbox to 618pt{\hsize6pc\vfill\vbox{\hsize6pc{\special{color push gray .15}{\affilfont\@affil\par}\vskip1pt\rule{6pc}{.5pt}\par\vskip1pt{\corfont\@authornote\par}\rule{6pc}{.5pt}\par{{\corfont%
    \ifx\@received\@empty\else{\bfseries Received:}\ \@received\fi%
    \ifx\@revised\@empty\else;\ {\bfseries Revised:}\ \@revised\fi% 
     \ifx\@accepted\@empty\else;\ {\bfseries Accepted:}\ \@accepted\fi\par}}\rule{6pc}{.5pt}\special{color pop}}}}}}%
  \colorbox{dsubhdcolor}{\revsptitlefont \@subhead\par}\vskip9pt%
  {\titlefont \@title \par}%
    \vskip 9pt%
    {\authorfont\@author\par\vskip12pt}%
    \ifx\@abstract\@empty\else\vskip10pt{\abstractfont\@abstract\par}\fi%
    \ifx\@keywords\@empty\else\vskip9pt{\abstractfont{\keywordheadfont\bfseries Keywords:\space}\@keywords\par}\vskip4pt\fi%
  \egroup
  \vskip 1.5em}
%%
\setcounter{secnumdepth}{0}
\def\@seccntformat#1{\csname the#1\endcsname\hspace*{4.5pt}}
\newcounter {part}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thesection {\@arabic\c@section.}
\renewcommand\thesubsection   {\thesection\@arabic\c@subsection.}
\renewcommand\thesubsubsection{\thesubsection\@arabic\c@subsubsection.}
\renewcommand\theparagraph    {\thesubsubsection\@arabic\c@paragraph.}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\part{%
   \if@noskipsec \leavevmode \fi
   \par
   \addvspace{4ex}%
   \@afterindentfalse
   \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >\m@ne
       \Large\bfseries \partname\nobreakspace\thepart
       \par\nobreak
     \fi
     \huge \bfseries #2%
     \markboth{}{}\par}%
    \nobreak
    \vskip 3ex
    \@afterheading}
\def\@spart#1{%
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \huge \bfseries #1\par}%
     \nobreak
     \vskip 3ex
     \@afterheading}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-20pt \@plus -1pt \@minus -.5pt}%
                                   {4pt}%
                                   {\sectionfont}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-20pt \@plus -1pt \@minus -.5pt}%
                                     {4pt}%
                                     {\subsectionfont}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-12pt \@plus -1pt \@minus -.5pt}%
                                     {0.0001pt}%
                                     {\subsubsectionfont}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {12pt \@plus -1pt \@minus -.5pt}%%
                                    {-.5em}%
                                    {\paragraphfont}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {12pt \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\subparagraphfont}}
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #6{%
        \@hangfrom{\hskip #3\relax\@svsec}%
          \interlinepenalty \@M #8\@@par}%
    \endgroup
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth \else
        \protect\numberline{\csname the#1\endcsname}%
      \fi
      #7}%
  \else
    \def\@svsechd{%
      #6{\hskip #3\relax
      \@svsec #8.}%
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #7}}%
  \fi
  \@xsect{#5}}

\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\if@titlepage
  \newenvironment{abstract}{%
      \titlepage
      \null\vfil
      \@beginparpenalty\@lowpenalty
      \begin{center}%
        \bfseries \abstractname
        \@endparpenalty\@M
      \end{center}}%
     {\par\vfil\null\endtitlepage}
\else
  \newenvironment{abstract}{%
      \if@twocolumn
        \section*{\abstractname}%
      \else
        \small
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi}
\fi
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{%
\setlength{\leftmargin}{12pt}%
\setlength{\rightmargin}{12pt}}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\newcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\@Alph\c@section}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand \theequation {\@arabic\c@equation}
\newcounter{figure}
\renewcommand \thefigure {\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\nobreakspace\thefigure}
\newenvironment{figure}
               {\def\@captype{figure}\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\def\@captype{figure}\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\nobreakspace\thetable}
\newenvironment{table}
               {\def\@captype{table}\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\def\@captype{table}\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{2\p@}
\setlength\belowcaptionskip{0\p@}
\def\FigName{figure}
\long\def\@makecaption#1#2{%
 \ifx\FigName\@captype%
    \vskip\abovecaptionskip%
    {\figcapfont{\bfseries#1.}\space#2\par}%
 \else%
    {\tablecapfont{\bfseries#1.}\space#2\par}
    \vskip\belowcaptionskip%
 \fi}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{3}
\newcommand\tableofcontents{%
    \section*{\contentsname
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \@starttoc{toc}%
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty\@secpenalty
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
       \if@compatibility
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
      \fi
    \endgroup
  \fi}
\newcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus\p@}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\newcommand\listoffigures{%
    \section*{\listfigurename}%
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}%
    \@starttoc{lof}%
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \section*{\listtablename}%
      \@mkboth{%
          \MakeUppercase\listtablename}%
         {\MakeUppercase\listtablename}%
    \@starttoc{lot}%
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\section*{\refname}\bibfont%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \twocolumn[\section*{\indexname}]%
                \@mkboth{\MakeUppercase\indexname}%
                        {\MakeUppercase\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \columnseprule \z@
                \columnsep 35\p@
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{}
\def\@makefnmark{\textsuperscript{{\normalfont\@thefnmark}}}%
\newskip\@footmax
\def\footmax#1{%
  \setbox0\hbox{#1}%
  \global\@footmax\wd0}
\footmax{1}
\newcommand\@makefntext[1]{%
    \parindent0pt%
    \noindent
    \advance\@footmax by 8.5pt%
    \leavevmode\@hangfrom{\hbox to \@footmax{\@makefnmark\hfill}}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\refname{REFERENCES}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{ABSTRACT}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{15pt}%5mm
\setlength\columnseprule{0\p@}
\pagestyle{headings}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi
%%%%%%%%%%%%%%%%%% Aptara Corp %%%%%%%%%%%%
\def\@ixhpt{9.5}
\def\@xiihpt{12.5}

\renewcommand\normalsize{%
   \@setfontsize\normalsize{10}{12}
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
\normalsize
\newcommand\small{%
   \@setfontsize\small\@ixpt{11}%
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand\footnotesize{%
   \@setfontsize\footnotesize\@viiipt{11}%
   \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus\p@
   \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 3\p@ \@plus\p@ \@minus\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
\newcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
\newcommand\large{\@setfontsize\large\@xiipt{14}}
\newcommand\Large{\@setfontsize\Large\@xivpt{18}}
\newcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
\newcommand\huge{\@setfontsize\huge\@xxpt{25}}
\newcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}
\setlength\parindent{12\p@}
\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}
\setlength\headheight{12\p@}
\setlength\headsep   {14\p@}
\setlength\topskip   {10\p@}
\setlength\footskip{13\p@}
\if@compatibility \setlength\maxdepth{4\p@} \else
\setlength\maxdepth{.5\topskip} \fi
\if@compatibility
  \if@twocolumn
    \setlength\textwidth{410\p@}
  \else
    \setlength\textwidth{345\p@}
  \fi
\else
  \setlength\@tempdima{\paperwidth}
  \addtolength\@tempdima{-2in}
  \setlength\@tempdimb{345\p@}
  \if@twocolumn
    \ifdim\@tempdima>2\@tempdimb\relax
      \setlength\textwidth{2\@tempdimb}
    \else
      \setlength\textwidth{\@tempdima}
    \fi
  \else
    \ifdim\@tempdima>\@tempdimb\relax
      \setlength\textwidth{\@tempdimb}
    \else
      \setlength\textwidth{\@tempdima}
    \fi
  \fi
\fi
\if@compatibility\else
  \@settopoint\textwidth
\fi
\if@compatibility
  \setlength\textheight{43\baselineskip}
\else
  \setlength\@tempdima{\paperheight}
  \addtolength\@tempdima{-2in}
  \addtolength\@tempdima{-1.5in}
  \divide\@tempdima\baselineskip
  \@tempcnta=\@tempdima
  \setlength\textheight{\@tempcnta\baselineskip}
\fi
\addtolength\textheight{\topskip}
\if@twocolumn
 \setlength\marginparsep {10\p@}
\else
  \setlength\marginparsep{11\p@}
\fi
\setlength\marginparpush{5\p@}
\if@compatibility
  \if@twoside
     \setlength\oddsidemargin   {44\p@}
     \setlength\evensidemargin  {82\p@}
     \setlength\marginparwidth {107\p@}
  \else
     \setlength\oddsidemargin   {63\p@}
     \setlength\evensidemargin  {63\p@}
     \setlength\marginparwidth  {90\p@}
  \fi
  \if@twocolumn
     \setlength\oddsidemargin  {30\p@}
     \setlength\evensidemargin {30\p@}
     \setlength\marginparwidth {48\p@}
  \fi
\else
  \if@twoside
    \setlength\@tempdima        {\paperwidth}
    \addtolength\@tempdima      {-\textwidth}
    \setlength\oddsidemargin    {.4\@tempdima}
    \addtolength\oddsidemargin  {-1in}
    \setlength\marginparwidth   {.6\@tempdima}
    \addtolength\marginparwidth {-\marginparsep}
    \addtolength\marginparwidth {-0.4in}
  \else
    \setlength\@tempdima        {\paperwidth}
    \addtolength\@tempdima      {-\textwidth}
    \setlength\oddsidemargin    {.5\@tempdima}
    \addtolength\oddsidemargin  {-1in}
    \setlength\marginparwidth   {.5\@tempdima}
    \addtolength\marginparwidth {-\marginparsep}
    \addtolength\marginparwidth {-0.4in}
    \addtolength\marginparwidth {-.4in}
  \fi
  \ifdim \marginparwidth >2in
     \setlength\marginparwidth{2in}
  \fi
  \@settopoint\oddsidemargin
  \@settopoint\marginparwidth
  \setlength\evensidemargin  {\paperwidth}
  \addtolength\evensidemargin{-2in}
  \addtolength\evensidemargin{-\textwidth}
  \addtolength\evensidemargin{-\oddsidemargin}
  \@settopoint\evensidemargin
\fi
\if@compatibility
  \setlength\topmargin{27pt}
\else
  \setlength\topmargin{\paperheight}
  \addtolength\topmargin{-2in}
  \addtolength\topmargin{-\headheight}
  \addtolength\topmargin{-\headsep}
  \addtolength\topmargin{-\textheight}
  \addtolength\topmargin{-\footskip}     % this might be wrong!
  \addtolength\topmargin{-.5\topmargin}
  \@settopoint\topmargin
\fi
\setlength\footnotesep{6.65\p@}
\setlength{\skip\footins}{12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\floatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\intextsep   {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dblfloatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\@fptop{0\p@ \@plus 1fil}
\setlength\@fpsep{8\p@ \@plus 2fil}
\setlength\@fpbot{0\p@ \@plus 1fil}
\setlength\@dblfptop{0\p@ \@plus 1fil}
\setlength\@dblfpsep{8\p@ \@plus 2fil}
\setlength\@dblfpbot{0\p@ \@plus 1fil}
\setlength\partopsep{2\p@ \@plus 1\p@ \@minus 1\p@}
\def\@listi{\leftmargin\leftmargini
            \parsep 4\p@ \@plus2\p@ \@minus\p@
            \topsep 8\p@ \@plus2\p@ \@minus4\p@
            \itemsep4\p@ \@plus2\p@ \@minus\p@}
\let\@listI\@listi
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep    4\p@ \@plus2\p@ \@minus\p@
              \parsep    2\p@ \@plus\p@  \@minus\p@
              \itemsep   \parsep}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2\p@ \@plus\p@\@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   \topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}
%%
\global\setlength\paperwidth{8.5in}
\global\setlength\paperheight{11in}

\newdimen\typeheight\typeheight58.5pc%
\newdimen\typewidth\typewidth40.5pc%44pc%
\newlength\extramargin
\setlength\extramargin{30mm}
\global\setlength\textheight{56.208pc}
\global\setlength\textwidth{33.5pc}

\global\topmargin42.5pt
\oddsidemargin55.7756pt%3.6pc
\evensidemargin55.5pt

\setlength\headheight{0\p@}% 
\setlength\headsep   {32pt}% \typeheight - \textheight - \headheight
\setlength\topskip   {0\p@}
\setlength\splittopskip   {\topskip}
\addtolength\textheight{4pt}
\advance\textheight-22.3pt
%
% COLORS DEFINITIONS
%
%%
%\parskip5.75pt
%\raggedright%
\voffset-6.75pc
\hoffset-5.3pc
%%
\renewcommand{\sfdefault}{phv}
\renewcommand{\rmdefault}{ptm}
\usepackage{newtxmath}
\usepackage[scaled=0.87]{helvet}

\DeclareMathVersion{sans}
\DeclareMathVersion{boldsans}

\DeclareFontEncoding{LMS}{}{}
\DeclareFontSubstitution{LMS}{ntxsy}{m}{n}
\DeclareFontSubstitution{U}{ntxsfmia}{m}{it}

\SetSymbolFont{operators}       {sans}{\encodingdefault}{\sfdefault}{m}{n}
\SetSymbolFont{letters}         {sans}{OML}{ntxsfmi}{m}{it}
\SetSymbolFont{lettersA}        {sans}{U}{ntxsfmia}{m}{it}
\SetSymbolFont{largesymbolsTXA} {sans}{U}{ntxexa}{m}{n}
\SetMathAlphabet{\mathbf}       {sans}{T1}{phv}{b}{n}
\SetMathAlphabet{\mathit}       {sans}{OML}{ntxsfmi}{m}{it}

\SetSymbolFont{operators}       {boldsans}{\encodingdefault}{\sfdefault}{b}{n}
\SetSymbolFont{letters}         {boldsans}{OML}{ntxsfmi}{b}{n}
\SetSymbolFont{symbols}         {boldsans}{LMS}{ntxsy}{b}{n}
\SetSymbolFont{AMSm}            {boldsans}{U}{ntxsym}{b}{n}
\SetSymbolFont{symbolsC}        {boldsans}{U}{ntxsyc}{b}{n}
\SetSymbolFont{largesymbols}    {boldsans}{LMX}{ntxexx}{b}{n}
\SetSymbolFont{largesymbolsTXA} {boldsans}{U}{ntxexa}{b}{n}
\SetSymbolFont{lettersA}        {boldsans}{U}{ntxsfmia}{b}{it}
\SetMathAlphabet{\mathbf}       {boldsans}{T1}{phv}{b}{n}
\SetMathAlphabet{\mathit}       {boldsans}{OML}{ntxsfmi}{b}{it}
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Output Routine  %%%%%%%%%%%%%%%%%%%%%%%%%%
%%
\def\YES{yes}
\def\withinfirstpage{yes}%
\def\trimmarks{}%
\def\@outputpage{%
%\special{color push  cmyk 0 0 0 1.0}%
\begingroup           % the \endgroup is put in by \aftergroup
%\let\firstmark\botmark
  \let \protect \noexpand
  \global\let\@@if@newlist\if@newlist
  \global\@newlistfalse
  \@resetactivechars
  \@parboxrestore
  \shipout \vbox{%
    \set@typeset@protect
    \aftergroup \endgroup
    \aftergroup \set@typeset@protect
                                % correct? or just restore by ending
                                % the group?
  \if@specialpage
    \global\@specialpagefalse\@nameuse{ps@\@specialstyle}%
  \fi
  \if@twoside
    \ifodd\count\z@ \let\@thehead\@oddhead \let\@thefoot\@oddfoot
         \let\@themargin\oddsidemargin
    \else \let\@thehead\@evenhead
       \let\@thefoot\@evenfoot \let\@themargin\evensidemargin
    \fi
  \fi
  \reset@font
  \normalsize
  \baselineskip\z@skip \lineskip\z@skip \lineskiplimit\z@
    \@begindvi\trimmarks
    \vskip \topmargin
    \advance\@themargin\extramargin%
    \moveright\@themargin \vbox {%
      \setbox\@tempboxa \vbox to\headheight{%
        \vfil
        \color@hbox
          \normalcolor
          \hb@xt@\textwidth {%
            \let \label \@gobble
            \let \index \@gobble
            \let \glossary \@gobble %% 21 Jun 91
            \@thehead
            }%
        \color@endbox
        }%                        %% 22 Feb 87
      \dp\@tempboxa \z@
      \box\@tempboxa
      \vskip \headsep
      \box\@outputbox
      \baselineskip \footskip
      \color@hbox
        \normalcolor
        \hb@xt@\textwidth{%
          \let \label \@gobble
          \let \index \@gobble      %% 22 Feb 87
          \let \glossary \@gobble   %% 21 Jun 91
          \@thefoot
          }%
      \color@endbox
      }%
    }%
\ifx\withinfirstpage\YES\global\advance\textheight22.3pt\gdef\withinfirstpage{no}\fi%
\global \@colht \textheight
\stepcounter{page}%
%\special{color pop}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End Trimmarks  %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
\pagestyle{headings}
%%
\def\revsptitlefont{\fontfamily{\sfdefault}\fontsize{10}{18}\selectfont\mathversion{sans}}
\def\rntitlefont{\fontfamily{\sfdefault}\fontsize{24}{24}\selectfont\rightskip0pc plus1fill}
\def\foliofont{\fontfamily{\rmdefault}\fontsize{8}{10}\selectfont}
\def\rhfont{\fontfamily{\sfdefault}\fontsize{9}{10}\selectfont\raggedleft}
\def\titlefont{\fontfamily{\sfdefault}\fontsize{19}{23}\selectfont\bfseries\rightskip0pc plus1fill\mathversion{boldsans}}
\def\authorfont{\fontfamily{\sfdefault}\fontsize{13.5}{15}\selectfont\rightskip0pc plus1fill\mathversion{sans}}%
\def\affilfont{\reset@font\fontfamily{\sfdefault}\fontsize{8}{9}\selectfont\rightskip0pt plus1fill\mathversion{sans}}%
\def\corfont{\reset@font\fontfamily{\sfdefault}\fontsize{6.5}{9.5}\selectfont\rightskip0pc plus1fill}%
\def\historyfont{\reset@font\fontfamily{\sfdefault}\fontsize{6.5}{9.5}\selectfont\rightskip0pc plus1fill}%
\def\abstractfont{\fontfamily{\rmdefault}\fontsize{10}{12}\selectfont\raggedright}
\def\absheadfont{\fontfamily{\sfdefault}\fontsize{11.5}{13.5}\selectfont\bfseries}
\def\keywordfont{\fontfamily{\rmdefault}\fontsize{10}{12}\selectfont\rightskip6pc}
\def\abstractheadfont{\fontfamily{\sfdefault}\fontsize{10.5}{12}\selectfont\bfseries}
\def\keywordheadfont{\fontfamily{\rmdefault}\fontsize{11.5}{12}\selectfont\bfseries}
\def\rffont{\fontfamily{\rmdefault}\fontsize{7}{9}\selectfont}
\def\rhhfont{\fontfamily{\sfdefault}\fontsize{8}{10}\selectfont}
\def\fignotefont{\fontfamily{\rmdefault}\fontsize{6}{9}\selectfont}
\def\bibliofont{\fontfamily{\sfdefault}\fontsize{8}{11}\selectfont}
%%
\def\sectionfont{\fontfamily{\sfdefault}\fontsize{10.5}{12}\selectfont\bfseries\rightskip0pt plus1fill\mathversion{boldsans}}
\def\sectionfont{\fontfamily{\sfdefault}\fontsize{10.5}{12}\selectfont\bfseries\rightskip0pt plus1fill\mathversion{boldsans}}
\def\subsectionfont{\fontfamily{\sfdefault}\fontsize{10.5}{12}\selectfont\bfseries\rightskip0pt plus1fill\mathversion{boldsans}}
\def\subsubsectionfont{\fontfamily{\sfdefault}\fontsize{11.5}{12}\selectfont\mathversion{sans}}
\def\paragraphfont{\fontfamily{\sfdefault}\fontsize{10}{12}\selectfont\itshape\mathversion{sans}}%
\def\subparagraphfont{\fontfamily{\rmdefault}\fontsize{10}{12}\selectfont\scshape\mathversion{normal}}%
%%
\def\figcapfont{\fontfamily{\sfdefault}\fontsize{8}{10}\selectfont\mathversion{sans}}
\def\tablecapfont{\fontfamily{\sfdefault}\fontsize{9}{11}\selectfont\mathversion{sans}}
\def\tablefont{\fontfamily{\rmdefault}\fontsize{8}{11.5}\selectfont}
%%
\def\keywords#1{\gdef\@keywords{#1}}\keywords{}
\def\jelcode#1{\gdef\@jelcode{#1}}\jelcode{}
%%
\def\tnotemark#1{\textsuperscript{#1}}%%
%%
\newenvironment{tablenotes}{\list{}{\setlength{\labelsep}{0pt}% 
\setlength{\labelwidth}{0pt}%
\setlength{\leftmargin}{0pt}%
\setlength{\rightmargin}{0pt}%
\setlength{\topsep}{0pt}%
\setlength{\itemsep}{0pt}%
\setlength{\partopsep}{0pt}%
\setlength{\listparindent}{0em}%
\setlength{\parsep}{0pt}}%
\item\relax\fignotefont%
}{\endlist}%
%%
\def\tnote#1{\par\textsuperscript{#1}}
%%
\setlength\parindent{12\p@}%
\def\abstract#1{\gdef\@abstract{#1}}
%%
\renewcommand\abstract[2][]{%
  \gdef\@abstract{%
    \setlength{\parindent}{\z@}%
    {\reset@font\absheadfont#1}\par\vskip4.25pt% LN20feb2016
    {\abstractfont#2\par\vspace*{0\p@}}%
  }}%
%%
\def\endpage#1{\gdef\@endpage{#1}}\endpage{0}
\AtEndDocument{\immediate\write\@mainaux{\string\endpage{\thepage}}%
\immediate\write\@mainaux{\string\footmax{\thefootnote}}}%
\def\title#1{\gdef\@title{#1}}\title{}
\def\dhead#1{\gdef\@dhead{#1}}\dhead{}
\def\subhead#1{\gdef\@subhead{#1}}\subhead{}
\def\jvol#1{\gdef\@jvol{#1}}\jvol{}
\def\jnum#1{\gdef\@jnum{#1}}\jnum{}
\def\doi#1{\gdef\@doi{#1}}\doi{}
\def\jyear#1{\gdef\@jyear{#1}}\jyear{}
\def\received#1{\xdef\@received{#1}}
\def\revised#1{\xdef\@revised{#1}}
\def\accepted#1{\xdef\@accepted{#1}}
\newcount\affcnt
\def\affil#1{%
\global\advance\affcnt\@ne%
\expandafter\gdef\csname aff\romannumeral\the\affcnt\endcsname{#1}%
}
\newcount\authorcount
\def\author#1{\global\advance\authorcount\@ne%
\expandafter\gdef\csname author\romannumeral\the\authorcount\endcsname{#1}%
}
%%
\newcount\tempcount
\newcount\tempacount
%%
\newcount\autcpcnt
\def\authornote#1{%
\global\advance\autcpcnt\@ne%
\expandafter\gdef\csname authornote\romannumeral\the\autcpcnt\endcsname{#1}%
}
%%
\newcounter{corsp}
\def\@authornote{%
  \ifnum\autcpcnt>\z@%
    \tempcount\z@%
    \loop%
       \ifnum\autcpcnt>\z@%
       \advance\tempcount\@ne%
        \refstepcounter{corsp}%
        {\parindent0pt\textsuperscript{\fnsymbol{corsp}}\csname authornote\romannumeral\the\tempcount\endcsname.\space}%
        \advance\autcpcnt\m@ne%
    \repeat%
  \fi%
}
%%
\def\@affil{%
  \ifnum\affcnt>\z@%
    \tempcount\z@%
    \loop%
       \ifnum\affcnt>\z@%
       \advance\tempcount\@ne%
        {\parindent0pt\csname aff\romannumeral\the\tempcount\endcsname\ifnum\affcnt>\@ne;\par\fi}%
        \advance\affcnt\m@ne%
    \repeat%
  \fi%
}
\def\@author{%
  \ifnum\authorcount>\z@%
    \tempcount\z@%
    \tempacount=\authorcount%
    \loop%
    \ifnum\tempacount>\z@%
     \advance\tempcount\@ne%
      \noindent\csname author\romannumeral\the\tempcount\endcsname%
      \ifnum\authorcount=\tw@%
          \ifnum\tempacount=\@ne\else\space and\space\fi%
      \else%
        \ifnum\tempacount=\tw@%
           \unskip\space and\space%
        \else%
           \ifnum\tempacount=\@ne\else\unskip,\space\fi% 
        \fi%
      \fi%
    \advance\tempacount\m@ne%
  \repeat%
\fi%
}
%%
%%
%%
\def\ps@plain{%
   \def\@oddhead{\vbox{\vspace*{-1pt}\hbox to \textwidth{\hspace*{-30.65mm}\vbox to 54pt{\hsize280pt\textcolor{dhdcolor}{\rntitlefont\@dhead}\vfill}\hfill\vbox to 54pt{\hsize180pt{\raggedleft\rhfont{\itshape National Science Review}\par
 \@jvol:\ \jobname,\ \@jyear\par%
doi:\ https://doi.org/\@doi\par}}}}}%
   \let\@evenhead\@oddhead
   \def\@oddfoot{\hbox to \textwidth{\hspace*{-30mm}\vbox to 5pt{
{\special{color push gray 0.60}\vrule width40.5pc height.5pt depth0pt\special{color pop}}\par%
{\hsize40.5pc{\rffont\copyright\  The Author(s)\ \@jyear.\ Published by Oxford University Press on behalf of China Science Publishing \& Media Ltd. This is an Open Access article distributed under the terms of the Creative Commons Attribution License (http://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.\par}}}}}%
   \def\@evenfoot{}%
   \let\@mkboth\@gobbletwo}
%%
\def\oddfolio{{\foliofont Page\ \thepage\ of \@endpage}}
\def\evenfolio{{\foliofont Page\ \thepage\ of \@endpage}}
%%
\def\@listi{%
   \leftmargin\leftmargini
   \labelwidth\leftmargini
   \advance\labelwidth-\labelsep
   \parsep   \z@ \@plus \p@
   \topsep  .5\baselineskip \@plus \p@ \@minus \p@
   \itemsep  \z@ \@plus \p@ \@minus \p@}
\let\@listI\@listi
\@listi
\def\@listii{%
   \leftmargin\leftmarginii
   \labelwidth\leftmarginii
   \advance\labelwidth-\labelsep
   \topsep  .5\baselineskip \@plus \p@ \@minus \p@
   \parsep   \z@ \@plus \p@
   \itemsep  \parsep}
\def\@listiii{%
   \leftmargin\leftmarginiii
   \labelwidth\leftmarginiii
   \advance\labelwidth-\labelsep
   \topsep     \z@ \@plus \p@ \@minus \p@
   \parsep     \z@ \@plus \p@
   \partopsep  \z@ \@plus \z@ \@minus \p@
   \itemsep    \topsep}
\def\@listiv{%
   \leftmargin\leftmarginiv
   \labelwidth\leftmarginiv
   \advance\labelwidth-\labelsep}
\def\@listv{%
   \leftmargin\leftmarginv
   \labelwidth\leftmarginv
   \advance\labelwidth-\labelsep}
\def\@listvi{%
   \leftmargin\leftmarginvi
   \labelwidth\leftmarginvi
   \advance\labelwidth-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\partopsep         \z@ \@plus 1\p@ \@minus 1\p@
\leftmargini     25\p@
\leftmarginii    22\p@
\leftmarginiii   13\p@
\leftmarginiv    17\p@
\leftmarginv     10\p@
\leftmarginvi    10\p@
\leftmargin\leftmargini
\labelsep         5\p@
\labelwidth\leftmargini
\advance\labelwidth-\labelsep
%
\def\itemize{\setlength\leftmargini{1em}\labelsep5pt%
  \ifnum \@itemdepth >\thr@@\@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \expandafter
    \list
      \csname\@itemitem\endcsname
      {\setlength\leftmarginii{1.1em}\itemsep0pt\ifnum \@listdepth >1\topsep0pt\else\topsep4.5pt\fi\parsep0pt\parskip0pt\def\makelabel##1{\hss\llap{##1}}}%
  \fi}
%%
\def\enumerate{\setlength\leftmargini{1.25em}%
  \ifnum \@enumdepth >\thr@@\@toodeep\else
    \advance\@enumdepth\@ne
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
      \expandafter
      \list
        \csname label\@enumctr\endcsname
        {\setlength\leftmarginii{1.7em}\itemsep0pt\ifnum \@listdepth >1\topsep0pt\else\topsep4.5pt\fi\parsep0pt\parskip0pt\usecounter\@enumctr\def\makelabel##1{\hss\llap{##1}}}%
  \fi}
%%
\newenvironment{unnumlist}{\list{}{\itemsep0pt\parsep0pt\parskip0pt\partopsep0pt\leftmargin12pt\itemindent-12pt}}{\endlist}%
%%% itemize/enumerate
\def\theenumi{\@arabic\c@enumi}
\def\labelenumi{\theenumi.}
\def\theenumii{\@alph\c@enumii}
\def\labelenumii{(\theenumii)}
\def\p@enumii{\theenumi}
\def\theenumiii{\@roman\c@enumiii}
\def\labelenumiii{\theenumiii.}
\def\p@enumiii{\theenumi(\theenumii)}
\def\theenumiv{\@Alph\c@enumiv}
\def\labelenumiv{\theenumiv.}
\def\p@enumiv{\p@enumiii\theenumiii}
\def\labelitemi{{\fontsize{10}{10}\selectfont\textbullet}}
\def\labelitemii{\normalfont\bfseries \textendash}
\def\labelitemiii{\textasteriskcentered}
\def\labelitemiv{\textperiodcentered}
%%
\def\figuresource#1{\par\vspace*{6pt}{\fignotefont{\it Source}:\ #1\par}}
\def\figurenote#1{{\fignotefont{\it Note}:\ #1\par}}
%%
\def\Hline{%
  \noalign{\ifnum0=`}\fi\hrule \@height .5pt \futurelet%\@height \arrayrulewidth
   \@tempa\@xhline}
%%
\def\toprule{\\[-7.75pt]\hline\\[-7.75pt]}
\def\colrule{\\[-8.75pt]\hline\\[-8pt]}
\def\botrule{\\[-8.5pt]\hline}
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bibliography Section   %%%%%%%%%%%%%%%%%%
%
\newdimen\bibindent
\newdimen\bibleftmargin
\newdimen\bibitemsep
\newdimen\bibparsep
\newdimen\bibtext
\newdimen\biblabelsep
\newdimen\bibleftmargini
%
%%%%% TRUE FOR CHAPTER TITLE AND FALSE FOR SECTION TITLE %%%%%%%%
\newif\ifbibchapter
\global\bibchaptertrue
%%%%%%%% Unnumberd References %%%%%%%%%%%
\setlength\bibleftmargin  {10pt}
\setlength\bibitemsep  {0pt}
\setlength\bibparsep  {0pt}
%%%%%%%% Numbered Within Squre Bracket %%%%%%%%%%%
\newif\ifnumsqure
\global\numsqurefalse
\setlength\biblabelsep  {10pt}
\setlength\bibleftmargini  {5.75pt}
%%%%%%%% Numbered With Dot %%%%%%%%%%%%%%%
\setlength\bibtext  {18pt}
%
\AtBeginDocument{%
\usepackage{cite}
\global\let\bibfont\bibliofont
\renewenvironment{thebibliography}[1]
     {\section*{\refname}\bibfont\labelsep         4\p@%
 %     \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
%%
\def\@biblabel#1{#1.}
\def\citepunct{,\penalty\citepunctpenalty}
%%
\renewcommand\newblock{}
 }
%%
%%
\DeclareRobustCommand{\BibTeX}{B\kern-.01em%
        {\sbox\z@ B%
         \vbox to\ht\z@{\vss\hbox{\check@mathfonts
                              \fontsize\sf@size\z@
                              \math@fontsfalse\selectfont
                              IB}%
                        }%
        }%
        \kern-.05em%
        \TeX}
%%
\sloppy
\usepackage{color}
\definecolor{dhdcolor}{cmyk}{0,1,1,0.40}
\definecolor{dsubhdcolor}{cmyk}{0.04,0.04,0.10,0}%%
\endinput
%%
%% End of file `nsr.cls'.
