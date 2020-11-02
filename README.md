# post-a-pdf
test: can I post a pdf

\subsection{Dataset Construction}
We constructed our dataset from Baidu Maps. However, the raw data suffers from the lack of description and a lot of noise. We implemented the data preprocessing with some tricks including machine learning and  filtering rules. 
In this appendix, we mainly discuss how we collect reviews and ground-truth description from noisy map data. We also report the practice and results of our feature engineer including category information, context information and reviews retrieve.

%As described in Section 5.1, our dataset is built on an online map service, and the details on our dataset are presented in Table \ref{tab:data}. We also supplement details on dataset construction in following subsubsections.
%construction in this section.


\subsubsection{Reviews\;Collection} The POI reviews are not only the input of our model but also the basis of the following feature engineer.
We started our dataset construction from the review collection. In this step, we adopted some simple denoising methods to improve the quality of the review data. Generally, we filtered out two kinds of original reviews: too short reviews and reviews with repeated meaningless phrases such as ``Great, Great, Great''. 
For the first case, we observed that too short review is more likely to be meaningless and noisy. We filtered out the reviews that are shorter than 40 tokens. For the second case, we calculated the number of words $n_1$ and the number of unique words $n_2$ in a review. The review would be filtered out if $n_2/n_1 \leq 0.6$.
For the filtered reviews, we annotated the word with a frequency lower than 100 as unk (unknown) word, and build a vocabulary with 12,007 words.
Then we filtered out the reviews that contained more than 5 unk words.
\par In this way, we got 43,225,977 reviews that cover 2,122,675 POI. We concatenated the reviews for each POI as input. With attention mechanism especially self-attention, the order of the input source is not critical to represent learning. It means the concatenation here is reasonable. 
To alleviate the computation problem and the degradation of generative model with long sequence input, we kept the number of reviews less than 6 and the total length shorter than 300 tokens in concatenating. 
\subsubsection{Description\;Extraction} %Since there are few existing POI descriptions, we also face the difficulty of lacking groundtruth. 
It is disappointing that most of POI data suffer from the lack of ground-truth description. We also observed that the POI descriptions from Pedia are too stiff for ground-truth and can only cover a few famous brands. Following the research conducted by Novgorodov et al.\cite{Novgorodov19}, which pointed that some high-quality reviews can be seen as description directly, we applied a heuristic approach which combined the machine learning model and some simple filtering rules to extract description from reviews.
\par In machine learning part, we collected a small labeled dataset $\mathcal{Z}$ by manual assessment. Dataset $\mathcal{Z}$, composed by 16,836 samples, is annotated whether a review can serve as a description. We implemented the discriminator model based on LSTM to learn the projection, following Novgorodov et al.\cite{Novgorodov19}
\par For filtering rules, we analyzed the distribution of the words in reviews and descriptions.
Table \ref{tab:words} shows some words with a large difference between the frequency in reviews and descriptions. We can find that ``very'' has a larger frequency because it tends to convey extreme emotions, while ``again'', ``satisfy'', ``feel'' and ``friend'' have a larger frequency in reviews because they tend to describe a personal experience rather than description. However, we expected the description to be objective, informative and vivid. So we adopted the following filtering rules.
\begin{itemize}
    \item Length control. We found that short reviews are not informative enough, while long reviews are more likely to tell a specific story of personal experience. Both of them can not serve as objective description. So we first filtered the reviews whose length is between 100 and 240 experimentally.
    \item Reviews that with extreme words. Most of the reviews with extreme words are just venting emotion and insulting the store. They can not help describe the POI. Specifically, we removed reviews contain ``very sick'', ``disgusting'' and other extreme phrases.
    \item Reviews that tell a story about personal experience. Different from reviews,  we expected the description to be objective rather than subjective. To this end, we filtered out reviews contain ``very like'', ``has(have) visited'', ``often'' and other phrases, since most of these reviews are too personal.
\end{itemize}
\par For those extracted descriptions, we also built a vocabulary with 9,973 words in the same way with reviews. Finally, we extracted 691,224 descriptions which cover 380,990 POIs with the combined method. We organized a manual evaluation on 1,000 random samples, and the precision of our extraction is 95.2\%. The reviews that selected as description were removed from reviews data then.

\subsubsection{Category} 
As discussed in section \ref{sec:encoder_2}, we also collected the category information for each POI. In our dataset, the number of categories $m$ is 88. The distribution of categories is reported in Table \ref{tab:category}

\subsubsection{Context Information} 
We extracted context tensors in the way described in section \ref{sec:encoder_3}.
In practice, we extracted the nearby map from a square area with side length $h=3km$. Then we divided the nearby map into $10\times10$ grids. Since there are 88 POI categories in our dataset, the shape of the final context tensor is $10\times10\times88$.
\subsubsection{retrieve} \label{sec:apx_5}
We retrieved the reviews of similar POIs following the method in section \ref{sec:transfer_1}.
For co-query graph construction, we collected the query data from 2018-08-01 to 2018-08-15 in Baidu Maps, and set the time interval $\delta = 30min$. We also set the threshold values $\theta_l = 0.2km^{-1}$, $\theta_q = 50$ in practice.
However, not all POIs have enough similar POIs for retrieve. Given the similarity graph, we use the coverage rate to measure how many POIs have 3 or more similar POIs. We report that the coverage rates of brand similarity, location similarity and co-query similarity are 37.99\%, 98.53\% and 51.72\% respectively. The coverage rate is also consistent with the performance of the corresponding reconstruction channel (see Figure \ref{fig:trans_com}).
\begin{table}[h]
  \caption{Frequency of Words in Reviews and Descriptions}
  \label{tab:words}
  \begin{center}
  \begin{tabular}{ccl}
    \hline
    Word &  Reviews & Descriptions\\
    \hline
    again & 41.04\% & 15.78\%\\
    very &  30.83\% & 15.59\% \\
    satisfy & 13.96\% & 0.22\%\\
    feel & 11.00\% & 0.27\%\\
    often & 10.24\% & 0.20\%\\
    friend & 10.78\% & 0.99\%\\
    ... & ...&...  \\
  \hline
\end{tabular}
\end{center}
\end{table}
\begin{table}[h]
  \caption{Proportion of Different Categories}
  \label{tab:category}
  \begin{center}
  \begin{tabular}{ccl}
    \hline
    Tag & Proportion \\
    \hline
    Chinese restaurant & 25.64\%\\
    Fast-food restaurant  &  10.46\% \\
    Express Inn & 6.44\%\\
    Cake shop & 5.75\%\\
    Small store & 4.86\%\\
    Bath and massage & 2.56\%\\
    ... & ...  \\
  \hline
\end{tabular}
\end{center}
\end{table}
