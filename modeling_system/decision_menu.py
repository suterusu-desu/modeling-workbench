"""Compile compatible operation/target choices into one provider-neutral request."""


def factor_choices(options, instructions):
    """IDs/family keys stay local. Only explicit public descriptions go on wire."""
    if not options or len({o['id'] for o in options}) != len(options):
        raise ValueError('A nonempty menu with unique local IDs is required')
    groups = {}
    for option in options:
        if any(not isinstance(option.get(k), str) or not option[k].strip()
               for k in ('id', 'family', 'family_description', 'target_description')):
            raise ValueError('Explicit family and reviewed public descriptions required')
        group = groups.setdefault(option['family'], [])
        if group and group[0]['family_description'] != option['family_description']:
            raise ValueError('One consistent description per operation family')
        group.append(option)
    questions, families, criteria = {}, {}, {}
    for index, members in enumerate(groups.values()):
        alias = 'f' + str(index)
        criteria[alias] = members[0]['family_description']
        if len(members) == 1:
            families[alias] = {'single': members[0]['id']}
        else:
            question = 'target_' + alias
            targets = {'t' + str(i): row['id'] for i, row in enumerate(members)}
            questions[question] = {'type': 'choice', 'instructions':
                'If the operation is ' + members[0]['family_description'] +
                ', choose its target. ' + instructions,
                'criteria': {'t' + str(i): row['target_description'] for i, row in enumerate(members)}}
            families[alias] = {'question': question, 'targets': targets}
    operation_question = 'operation' if len(families) > 1 else None
    if operation_question:
        questions = {'operation': {'type': 'choice', 'instructions': instructions,
                                   'criteria': criteria}, **questions}
    if len(questions) > 12:
        raise ValueError('Menu exceeds bounded question count; narrow the actual candidate set')
    return {'questions': questions, 'mapping': {'families': families,
            'operation_question': operation_question,
            'fixed_family': next(iter(families)) if not operation_question else None}}


def resolve_choice(mapping, choices):
    """Only the selected family's conditional target can produce an operation."""
    question = mapping['operation_question']
    family = choices[question] if question else mapping['fixed_family']
    branch = mapping['families'][family]
    if 'single' in branch:
        return branch['single']
    return branch['targets'][choices[branch['question']]]
