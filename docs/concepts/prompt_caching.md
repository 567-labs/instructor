---
title: Understanding Prompt Caching for API Efficiency
description: Explore how prompt caching optimizes performance for API calls in OpenAI and Anthropic, enhancing efficiency and reducing costs.
---

## See Also

- [Caching](./caching.md) - General caching concepts
- [Cost Optimization](../examples/batch_job_oai.md) - Reduce API costs
- [Performance Optimization](../examples/sqlmodel.md#performance-optimization) - Performance best practices
- [Anthropic Integration](../integrations/anthropic.md) - Anthropic prompt caching support

# Prompt Caching

Prompt Caching is a feature that allows you to cache portions of your prompt, optimizing performance for multiple API calls with shared context. This helps to reduce cost and improve response times.

## Prompt Caching in OpenAI

OpenAI implements a prompt caching mechanism to optimize performance for API requests with similar prompts.

> Prompt Caching works automatically on all your API requests (no code changes required) and has no additional fees associated with it.

This optimization is especially useful for applications making multiple API calls with shared context, minimizing redundant processing and improving overall performance.

Prompt Caching is enabled for the following models:

- gpt-4o
- gpt-4.1-mini
- o1-preview
- o1-mini

Caching is based on prefix matching, so if you're using a system prompt that contains a common set of instructions, you're likely to see a cache hit as long as you move all variable parts of the prompt to the end of the message when possible.

## Prompt Caching in Anthropic

Prompt Caching is now generally available for Anthropic. This enables you to cache specific prompt portions, reuse cached content in subsequent calls, and reduce processed data per request.

??? note "Source Text"

    In the following example, we'll be using a short excerpt from the novel "Pride and Prejudice" by Jane Austen. This text serves as an example of a substantial context that might typically lead to slow response times and high costs when working with language models. You can download it manually [here](https://www.gutenberg.org/cache/epub/1342/pg1342.txt)

    ```
        _Walt Whitman has somewhere a fine and just distinction between "loving
    by allowance" and "loving with personal love." This distinction applies
    to books as well as to men and women; and in the case of the not very
    numerous authors who are the objects of the personal affection, it
    brings a curious consequence with it. There is much more difference as
    to their best work than in the case of those others who are loved "by
    allowance" by convention, and because it is felt to be the right and
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
    eminently quintessential of its author's works; and for this contention
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
    natural occasions provided by the false account of Darcy's conduct given
    by Wickham, and by the awkwardness (arising with equal naturalness) from
    the gradual transformation of Elizabeth's own feelings from positive
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
    seem to me the very masterpieces of Miss Austen's humour and of her
    faculty of character-creation--masterpieces who may indeed admit John
    Thorpe, the Eltons, Mrs. Norris, and one or two others to their company,
    but who, in one instance certainly, and perhaps in others, are still
    superior to them._
    _The characteristics of Miss Austen's humour are so subtle and delicate
    that they are, perhaps, at all times easier to apprehend than to
    express, and at any particular time likely to be differently
    apprehended by different persons. To me this humour seems to possess a
    greater affinity, on the whole, to that of Addison than to any other of
    the numerous species of this great British genus. The differences of
    scheme, of time, of subject, of literary convention, are, of course,
    obvious enough; the difference of sex does not, perhaps, count for much,
    for there was a distinctly feminine element in "Mr. Spectator," and in
    Jane Austen's genius there was, though nothing mannish, much that was
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
    nineteenth; and no doubt Miss Austen's principles, as well as her heart,
    would have shrunk from such things as the letter from the unfortunate
    husband in the_ Spectator, _who describes, with all the gusto and all the
    innocence in the world, how his wife and his friend induce him to play
    at blind-man's-buff. But another_ Spectator _letter--that of the damsel
    of fourteen who wishes to marry Mr. Shapely, and assures her selected
    Mentor that "he admires your_ Spectators _mightily"--might have been
    written by a rather more ladylike and intelligent Lydia Bennet in the
    days of Lydia's great-grandmother; while, on the other hand, some (I
    think unreasonably) have found "cynicism" in touches of Miss Austen's
    own, such as her satire of Mrs. Musgrove's self-deceiving regrets over
    her son. But this word "cynical" is one of the most misused in the
    English language, especially when, by a glaring and gratuitous
    falsification of its original sense, it is applied, not to rough and
    snarling invective, but to gentle and oblique satire. If cynicism means
    the perception of "the other side," the sense of "the accepted hells
    beneath," the consciousness that motives are nearly always mixed, and
    that to seem is not identical with to be--if this be cynicism, then
    every man and woman who is not a fool, who does not care to live in a
    fool's paradise, who has knowledge of nature and the world and life, is
    a cynic. And in that sense Miss Austen certainly was one. She may even
    have been one in the further sense that, like her own Mr. Bennet, she
    took an epicurean delight in dissecting, in displaying, in setting at
    work her fools and her mean persons. I think she did take this delight,
    and I do not think at all the worse of her for it as a woman, while she
    was immensely the better for it as an artist.
    ```

```python
import instructor
from pydantic import BaseModel


class Character(BaseModel):
    name: str
    description: str


# Note: For testing this example locally, create a book.txt file with content like:
# Sample book.txt content:
# "Pride and Prejudice by Jane Austen
#
# It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
# However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is
# so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or
# other of their daughters..."
book = """
Pride and Prejudice by Jane Austen

It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is
so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or
other of their daughters...
"""

# Uncomment to read from an actual file instead of using the sample text above
# with open("./book.txt") as f:
#     book = f.read()

client = instructor.from_provider("anthropic/claude-3-5-sonnet-20240620")

resp, completion = client.create_with_completion(
    model="claude-3-5-sonnet-20240620",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<book>" + book + "</book>",
                    "cache_control": {"type": "ephemeral"},  # (1)!
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

print(completion)
# Message(
#     id='msg_01QcqjktYc1PXL8nk7y5hkMV',
#     content=[
#         ToolUseBlock(
#             id='toolu_019wABRzQxtSbXeuuRwvJo15',
#             input={
#                 'name': 'Jane Austen',
#                 'description': 'A renowned English novelist of the early 19th century, known for her wit, humor, and keen observations of human nature. She is the author of
# several classic novels including "Pride and Prejudice," "Emma," "Sense and Sensibility," and "Mansfield Park." Austen\'s writing is characterized by its subtlety, delicate touch,
# and ability to create memorable characters. Her work often involves social commentary and explores themes of love, marriage, and societal expectations in Regency-era England.'
#             },
#             name='Character',
#             type='tool_use'
#         )
#     ],
#     model='claude-3-5-sonnet-20240620',
#     role='assistant',
#     stop_reason='tool_use',
#     stop_sequence=None,
#     type='message',
#     usage=Usage(cache_creation_input_tokens=2777, cache_read_input_tokens=0, input_tokens=30, output_tokens=161)
# )
```

1. Anthropic requires that you explicitly pass in the `cache_control` parameter to indicate that you want to cache the content.

!!! Warning "Caching Considerations"

    **Minimum cache size**: For Claude Haiku, your cached content needs to be a minimum of 2048 tokens. For Claude Sonnet, the minimum is 1024 tokens.

**Benefits**: The cost of reading from the cache is 10x lower than if we were to process the same message again and enables us to execute our queries significantly faster.

We've written a more detailed blog on how to use the `create_with_completion` method [here](../blog/posts/anthropic-prompt-caching.md) to validate you're getting a cache hit with instructor.
