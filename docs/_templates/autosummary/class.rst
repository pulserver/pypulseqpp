{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

   {% block methods %}
   {% set public = methods | reject("eq", "__init__") | list %}
   {% if public %}
   .. rubric:: {{ _("Methods") }}

   .. autosummary::
      :nosignatures:
   {% for item in public %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
