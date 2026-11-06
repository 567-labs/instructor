---
authors:
- ivanleomk
categories:
- Anthropic
comments: true
date: 2024-09-14
description: Discover how prompt caching with Anthropic can improve response times
  and reduce costs for large context applications.
draft: false
tags:
- prompt caching
- Anthropic
- API optimization
- cost reduction
- latency improvement
---

# Why should I use prompt caching?

Developers often face two key challenges when working with large context - Slow response times and high costs. This is especially true when we're making multiple of these calls over time, severely impacting the cost and latency of our applications. With Anthropic's new prompt caching feature, we can easily solve both of these issues.

Since the new feature is still in beta, we're going to wait for it to be generally available before we integrate it into instructor. In the meantime, we've put together a quickstart guide on how to use the feature in your own applications.

<!-- more -->

!!! warning "Caching Limitations"

    There are a few important limitations to be aware of when using prompt caching:

    - **Minimum cache size**: For Claude Haiku, your cached content needs to be a minimum of 2048 tokens. For Claude Sonnet, the minimum is 1024 tokens.

    - **Tool definitions**: Currently, tool definitions cannot be cached. However, support for caching tool definitions is planned for a future update.

    - **Upgrade Anthropic**: You must upgrade to Anthropic version `0.34.0` or later to use prompt caching. Make sure that you're using the latest version of the Anthropic SDK.

    Keep these limitations in mind when implementing prompt caching in your applications.

??? note "Source Text"

    In the following example, we'll be using a short excerpt from the novel "Pride and Prejudice" by Jane Austen. This text serves as an example of a substantial context that might typically lead to slow response times and high costs when working with language models. You can download it manually [here](https://www.gutenberg.org/cache/epub/1342/pg1342.txt)

    ```
        _Walt Whitman has somewhere a fine and just distinction between “loving
    by allowance” and “loving with personal love.” This distinction applies
    to books as well as to men and women; and in the case of the not very
    numerous authors who are the objects of the personal affection, it
    brings a curious consequence with it. There is much more difference as
    to their best work than in the case of those others who are loved “by
    allowance” by convention, and because it is felt to be the right and
    proper thing to love them. And in the sect--fairly large and yet
    unusually choice--of Austenians or Janites, there would probably be
    found partisans of the claim to primacy of almost every one of the
    novels. To some the delightful freshness and humour of_ Northanger
    Abbey, _its completeness, finish, and_ entrain, _obscure the undoubted
    critical facts that its scale is small, and its scheme, after all, that
    of burlesque or parody, a kind in which the first rank is reached with
    difficulty._ Persuasion, _relatively faint in tone, and not enthralling
    in interest, has devotees who exalt above all the others its exquisite
    delicacy and keeping. The catastrophe of_ Mansfield Park _is admittedly
    theatrical, the hero and heroine are insipid, and the author has almost
    wickedly destroyed all romantic interest by expressly admitting that
    Edmund only took Fanny because Mary shocked him, and that Fanny might
    very likely have taken Crawford if he had been a little more assiduous;
    yet the matchless rehearsal-scenes and the characters of Mrs. Norris and
    others have secured, I believe, a considerable party for it._ Sense and
    Sensibility _has perhaps the fewest out-and-out admirers; but it does
    not want them._
    _I suppose, however, that the majority of at least competent votes
    would, all things considered, be divided between_ Emma _and the present
    book; and perhaps the vulgar verdict (if indeed a fondness for Miss
    Austen be not of itself a patent of exemption from any possible charge
    of vulgarity) would go for_ Emma. _It is the larger, the more varied, the
    more popular; the author had by the time of its composition seen rather
    more of the world, and had improved her general, though not her most
    peculiar and characteristic dialogue; such figures as Miss Bates, as the
    Eltons, cannot but unite the suffrages of everybody. On the other hand,
    I, for my part, declare for_ Pride and Prejudice _unhesitatingly. It
    seems to me the most perfect, the most characteristic, the most
    eminently quintessential of its author’s works; and for this contention
    in such narrow space as is permitted to me, I propose here to show
    cause._
    _In the first place, the book (it may be barely necessary to remind the
    reader) was in its first shape written very early, somewhere about 1796,
    when Miss Austen was barely twenty-one; though it was revised and
    finished at Chawton some fifteen years later, and was not published till
    1813, only four years before her death. I do not know whether, in this
    combination of the fresh and vigorous projection of youth, and the
    critical revision of middle life, there may be traced the distinct
    superiority in point of construction, which, as it seems to me, it
    possesses over all the others. The plot, though not elaborate, is almost
    regular enough for Fielding; hardly a character, hardly an incident
    could be retrenched without loss to the story. The elopement of Lydia
    and Wickham is not, like that of Crawford and Mrs. Rushworth, a_ coup de
    théâtre; _it connects itself in the strictest way with the course of the
    story earlier, and brings about the denouement with complete propriety.
    All the minor passages--the loves of Jane and Bingley, the advent of Mr.
    Collins, the visit to Hunsford, the Derbyshire tour--fit in after the
    same unostentatious, but masterly fashion. There is no attempt at the
    hide-and-seek, in-and-out business, which in the transactions between
    Frank Churchill and Jane Fairfax contributes no doubt a good deal to the
    intrigue of_ Emma, _but contributes it in a fashion which I do not think
    the best feature of that otherwise admirable book. Although Miss Austen
    always liked something of the misunderstanding kind, which afforded her
    opportunities for the display of the peculiar and incomparable talent to
    be noticed presently, she has been satisfied here with the perfectly
    natural occasions provided by the false account of Darcy’s conduct given
    by Wickham, and by the awkwardness (arising with equal naturalness) from
    the gradual transformation of Elizabeth’s own feelings from positive
    aversion to actual love. I do not know whether the all-grasping hand of
    the playwright has ever been laid upon_ Pride and Prejudice; _and I dare
    say that, if it were, the situations would prove not startling or
    garish enough for the footlights, the character-scheme too subtle and
    delicate for pit and gallery. But if the attempt were made, it would
    certainly not be hampered by any of those loosenesses of construction,
    which, sometimes disguised by the conveniences of which the novelist can
    avail himself, appear at once on the stage._
    _I think, however, though the thought will doubtless seem heretical to
    more than one school of critics, that construction is not the highest
    merit, the choicest gift, of the novelist. It sets off his other gifts
    and graces most advantageously to the critical eye; and the want of it
    will sometimes mar those graces--appreciably, though not quite
    consciously--to eyes by no means ultra-critical. But a very badly-built
    novel which excelled in pathetic or humorous character, or which
    displayed consummate command of dialogue--perhaps the rarest of all
    faculties--would be an infinitely better thing than a faultless plot
    acted and told by puppets with pebbles in their mouths. And despite the
    ability which Miss Austen has shown in working out the story, I for one
    should put_ Pride and Prejudice _far lower if it did not contain what
    seem to me the very masterpieces of Miss Austen’s humour and of her
    faculty of character-creation--masterpieces who may indeed admit John
    Thorpe, the Eltons, Mrs. Norris, and one or two others to their company,
    but who, in one instance certainly, and perhaps in others, are still
    superior to them._
    _The characteristics of Miss Austen’s humour are so subtle and delicate
    that they are, perhaps, at all times easier to apprehend than to
    express, and at any particular time likely to be differently
    apprehended by different persons. To me this humour seems to possess a
    greater affinity, on the whole, to that of Addison than to any other of
    the numerous species of this great British genus. The differences of
    scheme, of time, of subject, of literary convention, are, of course,
    obvious enough; the difference of sex does not, perhaps, count for much,
    for there was a distinctly feminine element in “Mr. Spectator,” and in
    Jane Austen’s genius there was, though nothing mannish, much that was
    masculine. But the likeness of quality consists in a great number of
    common subdivisions of quality--demureness, extreme minuteness of touch,
    avoidance of loud tones and glaring effects. Also there is in both a
    certain not inhuman or unamiable cruelty. It is the custom with those
    who judge grossly to contrast the good nature of Addison with the
    savagery of Swift, the mildness of Miss Austen with the boisterousness
    of Fielding and Smollett, even with the ferocious practical jokes that
    her immediate predecessor, Miss Burney, allowed without very much
    protest. Yet, both in Mr. Addison and in Miss Austen there is, though a
    restrained and well-mannered, an insatiable and ruthless delight in
    roasting and cutting up a fool. A man in the early eighteenth century,
    of course, could push this taste further than a lady in the early
    nineteenth; and no doubt Miss Austen’s principles, as well as her heart,
    would have shrunk from such things as the letter from the unfortunate
    husband in the_ Spectator, _who describes, with all the gusto and all the
    innocence in the world, how his wife and his friend induce him to play
    at blind-man’s-buff. But another_ Spectator _letter--that of the damsel
    of fourteen who wishes to marry Mr. Shapely, and assures her selected
    Mentor that “he admires your_ Spectators _mightily”--might have been
    written by a rather more ladylike and intelligent Lydia Bennet in the
    days of Lydia’s great-grandmother; while, on the other hand, some (I
    think unreasonably) have found “cynicism” in touches of Miss Austen’s
    own, such as her satire of Mrs. Musgrove’s self-deceiving regrets over
    her son. But this word “cynical” is one of the most misused in the
    English language, especially when, by a glaring and gratuitous
    falsification of its original sense, it is applied, not to rough and
    snarling invective, but to gentle and oblique satire. If cynicism means
    the perception of “the other side,” the sense of “the accepted hells
    beneath,” the consciousness that motives are nearly always mixed, and
    that to seem is not identical with to be--if this be cynicism, then
    every man and woman who is not a fool, who does not care to live in a
    fool’s paradise, who has knowledge of nature and the world and life, is
    a cynic. And in that sense Miss Austen certainly was one. She may even
    have been one in the further sense that, like her own Mr. Bennet, she
    took an epicurean delight in dissecting, in displaying, in setting at
    work her fools and her mean persons. I think she did take this delight,
    and I do not think at all the worse of her for it as a woman, while she
    was immensely the better for it as an artist.
    ```

Let's first initialize our Anthropic client, this will be the same as what we've done before except we're now using the new `beta.prompt_caching` method.

```python
from instructor import Instructor, Mode, patch
from anthropic import Anthropic

client = Instructor(
    client=Anthropic(),
    create=patch(
        create=Anthropic().beta.prompt_caching.messages.create,
        mode=Mode.TOOLS,
    ),
    mode=Mode.TOOLS,
)
```

We'll then create a new `Character` class that will be used to extract out a single character from the text and read in our source text ( roughly 2856 tokens using the Anthropic tokenizer).

```python
with open("./book.txt") as f:
    book = f.read()


class Character(BaseModel):
    name: str
    description: str
```

Once we've done this, we can then make an api call to get the description of the character.

```python
for _ in range(2):
    resp, completion = client.create_with_completion(  # (1)!
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<book>" + book + "</book>",
                        "cache_control": {"type": "ephemeral"},  # (2)!
                    },
                    {
                        "type": "text",
                        "text": "Extract a character from the text given above",
                    },
                ],
            },
        ],
        response_model=Character,
        max_tokens=1000,
    )
    assert isinstance(resp, Character)

    print(completion.usage)  # (3)!
    print(resp)
```

1. Using the `create_with_completion` method we can get back both the structured response and the completion object
2. We set the `cache_control` parameter to "ephemeral" to tell Anthropic to cache the book content temporarily
3. We print out the usage information to monitor token consumption

You'll notice that the usage information is different than what we've seen before. This is because we're now using the `create_with_completion` method which returns both the structured response and the completion object. The completion object contains usage information which we can use to monitor token consumption.

When we run this, you'll notice that we get the following output.

```bash
PromptCachingBetaUsage(
    cache_creation_input_tokens=2856,
    cache_read_input_tokens=0,
    input_tokens=30,
    output_tokens=119
)

Character(
    name='Elizabeth Bennet',
    description="The protagonist of Jane Austen's novel Pride and Prejudice, who
undergoes a transformation from initially disliking Mr. Darcy to eventually falling
in love with him. The passage describes Elizabeth as a complex, nuanced character,
noting how her feelings towards Darcy evolve naturally over the course of the story."
)

PromptCachingBetaUsage(
    cache_creation_input_tokens=0,
    cache_read_input_tokens=2856,
    input_tokens=30,
    output_tokens=93
)

Character(
    name='Mrs. Norris',
    description='A character from Jane Austen\'s novel Mansfield Park, described as
having "matchless" scenes and being one of the characters that has secured a
considerable party of admirers for the novel.'
)
```

You'll notice that in the first request, we created `2856` tokens and in the second request, we read `2856` tokens.

In other words, `book_content` was cached after the first request and reused in the second request. When you have a larger context window, this can save you a significant amount of money and time because your requests will return a lot faster too.

This is the entire code for the example above.

```python
from instructor import Instructor, Mode, patch
from anthropic import Anthropic
from pydantic import BaseModel

client = Instructor(
    client=Anthropic(),
    create=patch(
        create=Anthropic().beta.prompt_caching.messages.create,
        mode=Mode.TOOLS,
    ),
    mode=Mode.TOOLS,
)


class Character(BaseModel):
    name: str
    description: str


with open("./book.txt") as f:
    book = f.read()

for _ in range(2):
    resp, completion = client.create_with_completion(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<book>" + book + "</book>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "Extract a character from the text given above",
                    },
                ],
            },
        ],
        response_model=Character,
        max_tokens=1000,
    )
    assert isinstance(resp, Character)
    print(completion.usage)
    print(resp)
```

## Related Documentation
- [Caching Strategies](../../concepts/caching.md) - General caching concepts
- [Anthropic Integration](../../integrations/anthropic.md) - Full Anthropic guide

## See Also
- [Anthropic Structured Outputs](structured-output-anthropic.md) - Use with caching
- [Response Caching](caching.md) - General caching strategies
- [Performance Monitoring](logfire.md) - Track cache performance