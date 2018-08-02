Shitty CW Generator
===================

> Generate CWs using an LSTM

**This project is just for fun.** Please don't take it too seriously or think
I'm trying to censor the world or remove humans from CWs, I'm just having
fun with RNNs and thought CWs would be a fun way to do so.

## How do I make this thing work?

¯\\_(ツ)_/¯

Maybe this'll work for you though. Message me on fedi or open an issue and I'll
help you if (when) this doesn't work.

Install some things:

1. [Install theano](deeplearning.net/software/theano/install.html). Good luck
   with this tbqh because it was hell for me. I think what ultimately worked for
   me was doing it in a conda venv (`conda create -p venv`) with Python 3.7
   *despite the fact that theano doesn't officially support python >= 3.6*.
   Beats me.
2. `pip install keras`. Don't use `conda install` because it'll give you an
   outdated keras *and you'll get obscure errors that mislead you*.
3. For the scraper: `pip install mastodon.py`

Scrape the data:

1. `cd data`
2. `./consent-scrape.py`
3. Wait for eternity to end. I hope you brought a book.

Learn from the data:

1. Be in the root directory. If you were just in data, then `cd ..`
2. `./lstm_seq2seq.py`. Add a `--help` to see a few arguments you can add.
3. Wait for eternity times three. Gosh, ML has such slow feedback loops :(

Use the data:

1. Use?? What do you mean? You realize this was **just for fun** right? Just
   laugh at the shitty results.

## Wait what kind of shitty magic is this made out of?

It's a dead simple single-layer LSTM. One encoder module, one decoder module. It
currently doesn't allow for any non-CW'd posts because otherwise it gets lazy
and never CWs (what a bad netizen! don't @ me). In the future I intend to add in
the un-CWd posts after it learns to CW in general.

Most of the code is straight lifted from
[a keras tutorial](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html).

## How I dealt with consent in the scraping

I literally just asked people to opt in. I made
[a post](https://anticapitalist.party/@cosine/100478734475112961)
saying "fave this to consent", put their names in a text file, and then I
download toots from each account

At some point I'd like to make a public list so that it's just as easy for
people to use that as the public timeline, but I'm not there yet because for
something public like that I'd wanna have an effective re-opt out and stuff.
Anyway, that's the \~future~.

